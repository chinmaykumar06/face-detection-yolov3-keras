# face-detection-yolov3-keras

## Overview of the project

![img](https://github.com/chinmaykumar06/face-detection-yolov3-keras/blob/main/outputs/test_yolov3.jpg)

The YOLOv3 (You Only Look Once) is a state-of-the-art, real-time object detection algorithm. The published model recognizes 80 different objects in images and videos. For more details, you can refer to this [paper](https://pjreddie.com/media/files/papers/YOLOv3.pdf).
The official [DarkNet GitHub](https://github.com/pjreddie/darknet) repository contains the source code for the YOLO versions mentioned in the papers, written in C. The repository provides a step-by-step tutorial on how to use the code for object detection.
Instead of developing this code from scratch, we can use a third-party implementation. There are many third-party implementations designed for using YOLO with Keras.

The basic steps follwed were
* For making the model the pre trained wight file which is trained on the [WIDER FACE: A Face Detection Benchmark](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/index.html) dataset from this [link](https://drive.google.com/file/d/1xYasjU52whXMLT5MtF7RCPQkV66993oR/view?usp=sharing) was used.
* Write a Keras model that has the right number and type of layers to match the downloaded model weights. The model architecture is called a [“DarkNet”](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg) and was originally loosely based on the VGG-16 model.
* Change the number of filters in the yolo layers to make it a classifier for single class detction, i.e. face.
* Build the model bby loading the model weights using the previously downloaded pretrained weights
* Perform the prediction.

## YOLOv3's architecture

![img](yolo-architecture.png)

Credit: [Ayoosh Kathuria](https://towardsdatascience.com/yolo-v3-object-detection-53fb7d3bfe6b)

https://github.com/experiencor/keras-yolo3/blob/master/yolo3_one_file_to_detect_them_all.py

## Prerequisites

* Tensorflow
* opencv-python
* opencv-contrib-python
* Numpy
* Keras
* Matplotlib
* Pillow
