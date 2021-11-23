import os
import random
import numpy as np
from csv import reader
import shutil
import logging


def load_from_csv(path: str, video_id_col: int = 0, set_col: int = 1):
    train, val, test = [], [], []
    with open(path, "r") as read_obj:
        csv_reader = reader(read_obj)
        for row in csv_reader:
            if row[set_col] == "training":
                train.append(row[video_id_col])
            if row[set_col] == "validation":
                val.append(row[video_id_col])
            if row[set_col] == "test":
                test.append(row[video_id_col])
    return train, val, test


def generate_random_sets(path: str, train_r: float, val_r: float):
    all_video_clips = [f for f in os.listdir(path)]
    n_videos = len(all_video_clips)
    random.shuffle(all_video_clips)

    indicies_for_splitting = [
        int(n_videos * train_r),
        int(n_videos * (train_r + val_r)),
    ]
    train, val, test = np.split(all_video_clips, indicies_for_splitting)

    return train, val, test


def create_directory(path: str):
    if not os.path.exists(path):
        logging.debug("Output folder created")
        os.makedirs(path)


def split_data(path: str, train: list, val: list, test: list):

    folders = [
        f"{path}/Myanmar_data/training_set/image/",
        f"{path}/Myanmar_data/test_set/image/",
        f"{path}/Myanmar_data/val_set/image/",
    ]

    for i in folders:
        create_directory(i)

    for t in train:
        shutil.copy(path, folders[0])
        print(f"move {t} to training folder")

    for v in val:
        shutil.copy(path, folders[1])
        print(f"move {v} to validation folder")

    for t in test:
        shutil.copy(path, folders[2])
        print(f"move {t} to test folder")
    print(len(train))
    print(len(val))
    print(len(test))


if __name__ == "__main__":
    path = "part2"
    # load_from_csv()
    train, val, test = generate_random_sets(path, 0.7, 0.1)
    split_data(path, train, val, test)
