import os
import cv2
from pytorchyolo import detect, models
from pytorchyolo.utils.utils import load_classes
import csv

# Load the YOLO model
model = models.load_model(
  "config/yolov3.cfg", 
  "weights/yolov3.weights")

classes = load_classes("data/coco.names")

root = "/work1/fbohy/Helmet/images/"

subfolder = ""
data = []
for dirs, subdir, files in os.walk(root):
    for file in files:
      current = dirs.split("/")[-1]
      if subfolder != current:
        # dump csv file
        subfolder = current
        print(data)
        data = []

      filename = os.path.join(dirs, file)
      # Load the image as a numpy array
      img = cv2.imread(filename)

      # Convert OpenCV bgr to rgb
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

      # Runs the YOLO model on the image 
      boxes = detect.detect_image(model, img)

      for b in boxes:
        data.append([file.split(".")[0], b[0], b[1], b[2], b[3], b[4], classes[int(b[5])]])

      # Output will be a numpy array in the following format:
      # [[x1, y1, x2, y2, confidence, class]]
