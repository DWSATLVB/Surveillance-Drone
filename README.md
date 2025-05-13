# Surveillance-Drone
Code for Surveillance Drones using ESP32 equipped with ESP cam to detect weapons using YOLOv5, custom AI model. 

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
