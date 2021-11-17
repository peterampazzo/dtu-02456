"""
This script loads all the pictures from the Myanmar dataset shared on DTU HPC 
and run PyTorch-YOLOv3. The bounding box coordinates will be stored in a csv file.
It is possible to specify which classes should be dumped in the csv file.
"""

import os
import cv2
from pytorchyolo import detect, models
from pytorchyolo.utils.utils import load_classes
import csv

img_root = "/work1/fbohy/Helmet/images/"
yolo_root = "pytorchyolo/"
csv_root = "output/"

if not os.path.exists(csv_root):
    os.makedirs(csv_root)

# Load the YOLO model
model = models.load_model(
    f"{yolo_root}config/yolov3.cfg", f"{yolo_root}weights/yolov3.weights"
)

classes = load_classes(f"{yolo_root}data/coco.names")
classes_allowed = [
    3
]  # https://github.com/eriklindernoren/PyTorch-YOLOv3/blob/master/data/coco.names

csv_header = ["frame", "x", "y", "w", "h", "label"]
subdir_l1 = [f.path for f in os.scandir(img_root) if f.is_dir()]

for l1 in subdir_l1:
    subdir_l2 = [f.path for f in os.scandir(l1) if f.is_dir()]
    for l2 in subdir_l2:
        current_folder = l2.split("/")[-1]
        data = []
        for f in os.scandir(l2):
            current_file = f.path.split("/")[-1].split(".")[0]

            # Load the image as a numpy array
            img = cv2.imread(f.path)

            # Convert OpenCV bgr to rgb
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Runs the YOLO model on the image
            boxes = detect.detect_image(model, img)

            for b in boxes:
                if int(b[5]) in classes_allowed:
                    data.append(
                        [current_file, b[0], b[1], b[2], b[3], b[4], classes[int(b[5])]]
                    )

            # Output will be a numpy array in the following format:
            # [[x1, y1, x2, y2, confidence, class]]

        # Dump into csv file all frame from the same directory
        with open(
            f"{csv_root}{current_folder}.csv", "w", encoding="UTF8", newline=""
        ) as output_file:
            writer = csv.writer(output_file)
            writer.writerow(csv_header)
            writer.writerows(data)
