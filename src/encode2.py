import os
import time
import copy
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from tqdm import tqdm
from utils import *
import argparse


parser = argparse.ArgumentParser()

parser.add_argument(
    "Project",
    metavar="project",
    type=str,
)
args = parser.parse_args()

project = args.Project

load_model = True
main_folder = "/work3/s203257"
origin = f"{main_folder}/{project}_raw/"
destination = f"{main_folder}/{project}_processed/"
model_path=f"save_model/Myanmar-encode.pt"

batch_size = 32
input_shape = (192, 192, 3)
train_ids = pd.read_csv(
    f"{main_folder}/{project}_annot/training_set.csv",
dtype={'frame_id': 'str'})
val_ids = pd.read_csv(
    f"{main_folder}/{project}_annot/val_set.csv",
dtype={'frame_id': 'str'})
test_ids = pd.read_csv(
    f"{main_folder}/{project}_annot/test_set.csv",
dtype={'frame_id': 'str'})

print("training data:", len(train_ids))
print("valid data:", len(val_ids))
print("test data:", len(test_ids))

dataset_sizes = {"train": len(train_ids), "val": len(val_ids), "test": len(test_ids)}
print(dataset_sizes)

"""Load loss function and metric"""
loss_fn = EncodeLoss()
encode_metric = EncodeMetric()

"""Position encode"""
names_to_labels = helmet_use_encode()

print(np.array(names_to_labels["DHelmetP1NoHelmet"]))


train_set = HelmetDataset(
    ids=train_ids,
    root_dir=f"{destination}training_set/image/",
    names_to_labels=names_to_labels,
    transform=transforms.Compose(
        [
            transforms.RandomRotation(10),
            transforms.RandomHorizontalFlip(),
            transforms.Resize((192, 192)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    video_frame_as_string=True
)

val_set = HelmetDataset(
    ids=val_ids,
    root_dir=f"{destination}val_set/image/",
    names_to_labels=names_to_labels,
    transform=transforms.Compose(
        [
            transforms.Resize((192, 192)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    video_frame_as_string=True
)

test_set = HelmetDataset(
    ids=test_ids,
    root_dir=f"{destination}test_set/image/",
    names_to_labels=names_to_labels,
    transform=transforms.Compose(
        [
            transforms.Resize((192, 192)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    video_frame_as_string=True
)

dataloaders = {
    "train": DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4),
    "val": DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4),
    "test": DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4),
}
print(dataloaders)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

map_location=torch.device('cpu')
model_ft = models.resnet34() if load_model else models.resnet34(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 10)
if load_model:
    model_ft.load_state_dict(torch.load(model_path,map_location=map_location))

model_ft = model_ft.to(device)

# Observe that all parameters are being optimized
# optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.0001)
optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.0001)

# Decay LR by a factor of 0.1 every 10 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)





def run_model(model):   

    # inputs, labels = train_set["image"], train_set["label"]
    # inputs, labels = inputs.type(torch.cuda.FloatTensor), labels.type(torch.cuda.FloatTensor)
    # inputs, labels = inputs.to(device), labels.to(device)


    # outputs = model(inputs)
    # # _, preds = torch.max(outputs, 1)
    # num_correct = encode_metric(outputs, labels)/len(train_ids)
    # # print(preds.shape,labels.shape)
    # loss = loss_fn(outputs, labels)/len(train_ids)


    # #loss = running_loss / dataset_sizes[phase]
    # #acc = running_corrects.double() / dataset_sizes[phase]


    # return num_correct
    model.eval()



    overall_loss = 0.0
    overall_acc = 0.0


    for phase in ["train","train", "val"]:

        running_loss = 0.0
        running_corrects = 0.0

        for i_batch, sample_batched in enumerate(dataloaders[phase]):
            inputs, labels = sample_batched["image"], sample_batched["label"]
            inputs, labels = inputs.type(torch.FloatTensor), labels.type(
                torch.FloatTensor
            )
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            # _, preds = torch.max(outputs, 1)
            num_correct = encode_metric(outputs, labels)
            # print(preds.shape,labels.shape)
            loss = loss_fn(outputs, labels)


            # statistics
            running_loss += loss.item() * inputs.size(0)
            # torch.sum(preds == labels.data)
            running_corrects += num_correct


        epoch_loss = running_loss / dataset_sizes[phase]
        epoch_acc = running_corrects.double() / dataset_sizes[phase]

        overall_loss += epoch_loss
        overall_acc += running_corrects

        print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

                

    return running_loss, running_corrects


savepath = f"save_model/{project}-encode.pt"
model_ft.load_state_dict(torch.load(savepath))
acc,loss = run_model(model_ft)
print("Loss: {:.4f} Acc: {:.4f}".format(loss, acc))




