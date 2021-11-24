"""
The code takes all the frames stored in the 'raw' folder and split it into train, test and
validation folders in a new folder called 'processed'.

It is possible to specify which project by command line.

Usage: python src/data_split.py <Project> # random
python src/data_split.py <Project> --csv # load sets from csv
"""


import os
import argparse
import random
import numpy as np
from csv import reader
from distutils.dir_util import copy_tree
import logging
import sys

logging.basicConfig(
    level=logging.DEBUG,
    format="[%(asctime)s] - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)


def load_from_csv(
    path: str, filename: str, directory: str, video_id_col: int = 0, set_col: int = 1
):
    file_path = os.path.join(path, filename)

    logging.info("Splitting data from csv file.")
    logging.debug(f"File loaded: {file_path}.")

    train, val, test = [], [], []
    with open(file_path, "r") as read_obj:
        csv_reader = reader(read_obj)
        for row in csv_reader:
            folder_path = os.path.join(path, directory, row[video_id_col])
            if row[set_col] == "training":
                train.append(folder_path)
            if row[set_col] == "validation":
                val.append(folder_path)
            if row[set_col] == "test":
                test.append(folder_path)
    return train, val, test


def generate_random_sets(path: str, train_r: float, val_r: float):
    all_video_clips = [os.path.join(path, f) for f in os.listdir(path)]
    n_videos = len(all_video_clips)
    random.shuffle(all_video_clips)

    indicies_for_splitting = [
        int(n_videos * train_r),
        int(n_videos * (train_r + val_r)),
    ]
    train, val, test = np.split(all_video_clips, indicies_for_splitting)

    logging.debug(f"Size training set: {len(train)}.")
    logging.debug(f"Size validation set: {len(val)}.")
    logging.debug(f"Size test set: {len(test)}.")

    return train, val, test


def create_directory(path: str):
    if not os.path.exists(path):
        logging.debug(f"Output folder {path} created.")
        os.makedirs(path)


def move_directories(folders: list, destination: str, set_name: str):
    for i in folders:
        directory = os.path.basename(os.path.normpath(i))
        copy_tree(i, os.path.join(destination, directory))
        logging.debug(f"Copying {i} to {set_name} folder.")


def split_data(destination: str, train: list, val: list, test: list):

    data = {
        "train": {
            "dest": f"{destination}/training_set/image/",
            "folders": train,
        },
        "test": {
            "dest": f"{destination}/test_set/image/",
            "folders": test,
        },
        "val": {"dest": f"{destination}/val_set/image/", "folders": val},
    }

    create_directory(destination)

    for i in data:
        create_directory(data[i]["dest"])
        move_directories(data[i]["folders"], data[i]["dest"], i)


if __name__ == "__main__":
    logging.info("Application started.")

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "Project",
        metavar="project",
        type=str,
    )
    parser.add_argument("--csv", action="store_true")
    args = parser.parse_args()

    project = args.Project
    main_folder = "/work3/s203257/"
    origin = f"{main_folder}/{project}_raw/"
    destination = f"{main_folder}/{project}_processed/"

    if args.csv:
        logging.info(f"Running project {project} on csv file.")
        train, val, test = load_from_csv(
            main_folder, f"{project}_annotation.csv", f"{project}_raw"
        )
    else:
        logging.info(f"Running project {project} with a random set.")
        train, val, test = generate_random_sets(origin, 0.7, 0.1)
    split_data(destination, train, val, test)
