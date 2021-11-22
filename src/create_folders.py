# go through the data split file and put the corresponding annotation from part_1 in a correct folder (train, test, validation)
import csv
import shutil
import os
import logging
import sys

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)

training, test, val = [], [], []

root = "/Users/rampazzo/GitHub/data"
filename = f"{root}/data_split.csv"

folders = [
    f"{root}/Myanmar_data/training_set/image/",
    f"{root}/Myanmar_data/test_set/image/",
    f"{root}/Myanmar_data/val_set/image/",
]

for i in folders:
    if not os.path.exists(i):
        os.makedirs(i)

with open(filename, "r") as csvfile:
    datareader = csv.reader(csvfile)
    next(datareader)
    for annot, split_type in datareader:
        try:
            folder_source = f"{root}/images/{annot}/"
            logging.debug(f"Folder: {folder_source} - {split_type}")

            if split_type == "training":
                training.append(annot)
                shutil.move(folder_source, folders[0])

            elif split_type == "test":
                test.append(annot)
                shutil.move(folder_source, folders[1])

            elif split_type == "validation":
                val.append(annot)
                shutil.move(folder_source, folders[2])
        except:
            logging.debug(f"Error with: {annot}.")
