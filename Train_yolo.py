import os

# Step 1: Clone YOLOv5 repository
os.system("git clone https://github.com/ultralytics/yolov5.git")
os.chdir("yolov5")

# Step 2: Install dependencies
os.system("pip install -r requirements.txt")

# Step 3: Mount Google Drive (if using Colab)
from google.colab import drive
drive.mount('/content/drive')

# Step 4: Train YOLOv5 model
train_cmd = (
    "python train.py --img 640 --batch 16 --epochs 50 "
    "--data /content/drive/MyDrive/yolo_dataset/dataset.yaml "
    "--weights yolov5s.pt --name custom_model"
)
os.system(train_cmd)

# Step 5: Copy trained model to Google Drive for easy access
os.system("cp runs/train/custom_model/weights/best.pt /content/drive/MyDrive/best.pt")

print("Training completed! Model saved as best.pt in Google Drive.")
