import cv2

# ESP32-CAM's IP address
ESP32_CAM_URL = ""

# Open video stream
cap = cv2.VideoCapture(ESP32_CAM_URL)

if not cap.isOpened():
    print("Error: Could not open video stream")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    cv2.imshow("ESP32-CAM", frame)

    # Press 'x' to exit
    if cv2.waitKey(1) & 0xFF == ord('x'):
        break

cap.release()
cv2.destroyAllWindows()
