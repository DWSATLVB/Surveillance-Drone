import torch
import cv2
import numpy as np

# Load the fine-tuned YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/exp/weights/best.pt') 

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

ESP32_CAM_URL = ""  # Replace with your ESP32-CAM IP
cap = cv2.VideoCapture(ESP32_CAM_URL)

if not cap.isOpened():
    print("Error: Could not open video stream")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
        
    results = model(frame)

    knives_and_guns = []
    for *box, conf, cls in zip(*results.xywh[0].tolist(), results.conf[0].tolist(), results.cls[0].tolist()):
        if cls in [0, 1]:  # Class 0 for knife, Class 1 for gun
            knives_and_guns.append((box, conf, cls))

    for box, conf, cls in knives_and_guns:
        x1, y1, x2, y2 = box 
        label = f"{model.names[int(cls)]}: {conf:.2f}"

        # Draw the rectangle and label
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("YOLOv5 Object Detection (Knives & Guns)", frame)

    # Exit on 'x'
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cap.release()
cv2.destroyAllWindows()
