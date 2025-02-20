import torch
import cv2
import numpy as np

model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/exp/weights/best.pt')  # Use your best-trained model

# Ensure the model uses GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)

image_path = 'test_image.jpg'  # Path to your image file
img = cv2.imread(image_path)

if img is None:
    print("Error: Could not load image.")
    exit()

results = model(img) 

knives_and_guns = []
for *box, conf, cls in zip(*results.xywh[0].tolist(), results.conf[0].tolist(), results.cls[0].tolist()):
    if cls in [0, 1]:  # Class 0 for knife, Class 1 for gun
        knives_and_guns.append((box, conf, cls))

for box, conf, cls in knives_and_guns:
    x1, y1, x2, y2 = box  # Bounding box coordinates
    label = f"{model.names[int(cls)]}: {conf:.2f}"
    
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv2.putText(img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

cv2.imshow("YOLOv5 Detection (Knives & Guns)", img)

cv2.imwrite("detected_image.jpg", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
