import os
import random
from os.path import isfile, join
import numpy as np

def split_data(path,train_r,val_r):
    
    # create the splits in arrays    
    all_video_clips = [f for f in os.listdir(path) ]
    print(all_video_clips)
    
    n_videos = len(all_video_clips)
    random.shuffle(all_video_clips)  
    
    indicies_for_splitting = [int(n_videos * train_r), int(n_videos * (train_r+val_r))]
    train, val, test = np.split(all_video_clips, indicies_for_splitting)
    
    # create the folders from the arrays 
    folders = [
    f"{path}/Myanmar_data/training_set/image/",
    f"{path}/Myanmar_data/test_set/image/",
    f"{path}/Myanmar_data/val_set/image/",
    ]
    
    # create the directories 
    #    TODO
    
    for t in train:
        shutil.move(path, folder[0])
        print(f'move {t} to training folder')
        
    for v in val:
        shutil.move(path, folder[1])
        print(f'move {v} to validation folder')
        
    for t in test:
        shutil.move(path, folder[2])
        print(f'move {t} to test folder')
    print(len(train))
    print(len(val))
    print(len(test))
        
split_data('part_2', 0.7,0.1)

