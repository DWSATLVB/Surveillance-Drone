import os

""" !!!!IF NOT INSTALLED DEPENDENCIES FOR YOLOv5 THEN DO NOT SKIP THIS STEP!!!! """

os.system("git clone https://github.com/ultralytics/yolov5.git")
os.chdir("yolov5")

os.system("pip install -r requirements.txt")

from google.colab import drive
drive.mount('/content/drive')

train_cmd = (
    "python train.py --img 640 --batch 16 --epochs 50 "
    "--data /content/drive/MyDrive/yolo_dataset/dataset.yaml "
    "--weights yolov5s.pt --name custom_model"
)
os.system(train_cmd)

os.system("cp runs/train/custom_model/weights/best.pt /content/drive/MyDrive/best.pt")

print("Training completed! Model saved as best.pt in Google Drive.")
