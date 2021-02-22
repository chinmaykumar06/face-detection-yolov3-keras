# load yolov3 model and perform object detection
# we will use a pre-trained model to perform object detection on an unseen photograph
from numpy import expand_dims
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import cv2
import numpy as np
from tensorflow.keras import backend as K
from matplotlib import pyplot
import os
import argparse
import sys


class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, objness = None, classes = None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.objness = objness
        self.classes = classes
        self.label = -1
        self.score = -1
        
    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)
        
        return self.label
    
    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]
        
        return self.score


def _sigmoid(x):
    return 1. / (1. + np.exp(-x))
    

def decode_netout(netout, anchors, obj_thresh, net_h, net_w):
    grid_h, grid_w = netout.shape[:2] # 0 and 1 is row and column 13*13
    nb_box = 3 # 3 anchor boxes
    netout = netout.reshape((grid_h, grid_w, nb_box, -1)) #13*13*3 ,-1
    nb_class = netout.shape[-1] - 5
    boxes = []
    netout[..., :2]  = _sigmoid(netout[..., :2])
    netout[..., 4:]  = _sigmoid(netout[..., 4:])
    netout[..., 5:]  = netout[..., 4][..., np.newaxis] * netout[..., 5:]
    netout[..., 5:] *= netout[..., 5:] > obj_thresh
    
    for i in range(grid_h*grid_w):
        row = i / grid_w
        col = i % grid_w
        for b in range(nb_box):
            # 4th element is objectness score
            objectness = netout[int(row)][int(col)][b][4]
            if(objectness.all() <= obj_thresh): continue
            # first 4 elements are x, y, w, and h
            x, y, w, h = netout[int(row)][int(col)][b][:4]
            x = (col + x) / grid_w # center position, unit: image width
            y = (row + y) / grid_h # center position, unit: image height
            w = anchors[2 * b + 0] * np.exp(w) / net_w # unit: image width
            h = anchors[2 * b + 1] * np.exp(h) / net_h # unit: image height
            # last elements are class probabilities
            classes = netout[int(row)][col][b][5:]
            box = BoundBox(x-w/2, y-h/2, x+w/2, y+h/2, objectness, classes)
            boxes.append(box)
    return boxes


def correct_yolo_boxes(boxes, image_h, image_w, net_h, net_w):
    new_w, new_h = net_w, net_h
    for i in range(len(boxes)):
        x_offset, x_scale = (net_w - new_w)/2./net_w, float(new_w)/net_w
        y_offset, y_scale = (net_h - new_h)/2./net_h, float(new_h)/net_h
        boxes[i].xmin = int((boxes[i].xmin - x_offset) / x_scale * image_w)
        boxes[i].xmax = int((boxes[i].xmax - x_offset) / x_scale * image_w)
        boxes[i].ymin = int((boxes[i].ymin - y_offset) / y_scale * image_h)
        boxes[i].ymax = int((boxes[i].ymax - y_offset) / y_scale * image_h)


def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b
    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2,x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2,x4) - x3

#intersection over union        
def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])
    intersect = intersect_w * intersect_h
    
    
    w1, h1 = box1.xmax-box1.xmin, box1.ymax-box1.ymin  
    w2, h2 = box2.xmax-box2.xmin, box2.ymax-box2.ymin
    
    #Union(A,B) = A + B - Inter(A,B)
    union = w1*h1 + w2*h2 - intersect
    return float(intersect) / union


def do_nms(boxes, nms_thresh):    #boxes from correct_yolo_boxes and  decode_netout
    if len(boxes) > 0:
        nb_class = len(boxes[0].classes)
    else:
        return
    for c in range(nb_class):
        sorted_indices = np.argsort([-box.classes[c] for box in boxes])
        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]
            if boxes[index_i].classes[c] == 0: continue
            for j in range(i+1, len(sorted_indices)):
                index_j = sorted_indices[j]
                if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_thresh:
                    boxes[index_j].classes[c] = 0


# load and prepare an image
def load_image_pixels(filename, shape):
    # load the image to get its shape
    image = load_img(filename) #load_img() Keras function to load the image .
    width, height = image.size
    # load the image with the required size
    image = load_img(filename, target_size=shape) # target_size argument to resize the image after loading
    # convert to numpy array
    image = img_to_array(image)
    # scale pixel values to [0, 1]
    image = image.astype('float32')
    image /= 255.0  #rescale the pixel values from 0-255 to 0-1 32-bit floating point values.
    # add a dimension so that we have one sample
    image = expand_dims(image, 0)
    return image, width, height


# get all of the results above a threshold
def get_boxes(boxes, labels, thresh):
    v_boxes, v_labels, v_scores = list(), list(), list()
    # enumerate all boxes
    for box in boxes:
        # enumerate all possible labels
        for i in range(len(labels)):
            # check if the threshold for this label is high enough
            if box.classes[i] > thresh:
                v_boxes.append(box)
                v_labels.append(labels[i])
                v_scores.append(box.classes[i]*100)
    
    return v_boxes, v_labels, v_scores


# draw all results
def draw_boxes(filename, v_boxes, v_labels, v_scores, output_dir):
    #load the image
    img = cv2.imread(filename)
    for i in range(len(v_boxes)):
        # retrieving the coordinates from each bounding box
        box = v_boxes[i]
        # get coordinates
        y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
        start_point = (x1, y1) 
        # Ending coordinate
        # represents the bottom right corner of rectangle 
        end_point = (x2, y2) 
        # Red color in BGR 
        color = (0, 0, 255) 
        # Line thickness of 2 px 
        thickness = 2
        # font 
        font = cv2.FONT_HERSHEY_PLAIN 
        # fontScale 
        fontScale = 1.5
        #create the shape
        img = cv2.rectangle(img, start_point, end_point, color, thickness) 
        # draw text and score in top left corner
        label = "%s (%.3f)" % (v_labels[i], v_scores[i])
        img = cv2.putText(img, label, (x1,y1), font,  
                   fontScale, color, thickness, 2)
        text = "no.of faces detected faces: %s" % (len(v_boxes))
        img = cv2.putText(img, text, (10,40), font,  
                   fontScale, (255,0,0), thickness, 2)
    # show the plot
    output = filename.rsplit("/")[1].rsplit(".")[0]+'_yolov3.jpg'
    #save the image
    cv2.imwrite("./"+os.path.join(output_dir,output),img)
    print(filename)
    cv2.imshow("yolov3",img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def img_blur(filename, v_boxes, v_labels, v_scores, output_dir):
    img = cv2.imread(filename)
    rows, cols = img.shape[0], img.shape[1]
    blurred_img = cv2.GaussianBlur(img, (201, 201), 0)
    mask = np.zeros((rows, cols, 3), dtype=np.uint8)
    
    for i in range(len(v_boxes)):
        if (not v_boxes):
            x1,y1 = 0,0
            x2,y2 = 0,0
        else:
            box = v_boxes[i]
            y1, x1, y2, x2 = box.ymin, box.xmin, box.ymax, box.xmax
        mask = cv2.rectangle(mask, (x1, y1), (x2, y2), (255, 255, 255), -1)
    out = np.where(mask==np.array([255, 255, 255]), img, blurred_img)
    output = filename.rsplit("/")[1].rsplit(".")[0]+"_blur.jpg"
    cv2.imwrite("./"+os.path.join(output_dir,output), out) 
    cv2.imshow("img_blur",out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, default='',
                        help='path to image file')
    parser.add_argument('--output-dir', type=str, default='outputs/',
                        help='path to the output directory')
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        print('==> Creating the {} directory...'.format(args.output_dir))
        os.makedirs(args.output_dir)
    else:
        print('==> Skipping create the {} directory...'.format(args.output_dir))
    return args

    
def _main():
    # Get the arguments
    args = get_args()
    if args.image:
        if not os.path.isfile(args.image):
            print("[!] ==> Input image file {} doesn't exist".format(args.image))
            sys.exit(1)
    
    # define the anchors
    anchors = [[116,90, 156,198, 373,326], [30,61, 62,45, 59,119], [10,13, 16,30, 33,23]]  

    # define the probability threshold for detected objects
    class_threshold = 0.6
    #define class
    labels = ["face"]

    photo_filename = args.image
    output_dir = args.output_dir
    input_w, input_h = 416, 416
    # load and prepare image
    image, image_w, image_h = load_image_pixels(photo_filename, (input_w, input_h))

    # load yolov3 model
    model = load_model('model.h5')
    yhat = model.predict(image)
    # summarize the shape of the list of arrays

    boxes = list()
    for i in range(len(yhat)):
        # decode the output of the network
        boxes += decode_netout(yhat[i][0], anchors[i], class_threshold, input_h, input_w)
        
    # correct the sizes of the bounding boxes for the shape of the image
    correct_yolo_boxes(boxes, image_h, image_w, input_h, input_w)

    # suppress non-maximal boxes
    do_nms(boxes, 0.5)  #Discard all boxes with pc less or equal to 0.5

    # get the details of the detected objects
    v_boxes, v_labels, v_scores = get_boxes(boxes, labels, class_threshold)

    # draw what we found
    draw_boxes(photo_filename, v_boxes, v_labels, v_scores, output_dir)
    
    #blur the rest of image leaving the faces
    img_blur(photo_filename, v_boxes, v_labels, v_scores, output_dir)

    K.clear_session() 


if __name__ == "__main__":
    _main()








