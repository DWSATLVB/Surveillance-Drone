# Surveillance-Drone
Code for Surveillance Drones using ESP32 to detect weapons

## Description
This AI model is a custom-trained version of YOLOv5 (You Only Look Once version 5), a state-of-the-art object detection model. It __will be__ fine-tuned specifically to identify knives and guns from images or video streams. The model uses deep learning techniques to analyze the contents of the input data (such as images from a camera) and accurately detect and classify objects based on pre-trained weights. 

## Dependencies
#### **1. numpy**
#### **2. cv2**
#### **3. PyTorch**
#### **4. torchvision**
#### **5. YOLOv5 dependencies (via requirements.txt) - Additional libraries needed specifically for YOLOv5 model handling.**

## Installation Guide 
#### **1. For dependencies mentioned above (except YOLOv5)**
```bash
pip install numpy opencv-python torch torchvision
```
#### **2. Install YOLOv5 Dependencies**
```bash
pip install -r https://raw.githubusercontent.com/ultralytics/yolov5/main/requirements.txt
```
## Note 
To fine tune the model to detect certain objects requires customized dataset and cannot be done directly by YOLOv5. 
