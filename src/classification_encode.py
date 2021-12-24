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
import logging
import sys
from pyhocon import ConfigFactory

conf = ConfigFactory.parse_file("app.conf")

logging.basicConfig(
    level=logging.getLevelName(conf["logging"]["level"]),
    format="[%(asctime)s] - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    stream=sys.stdout,
)

parser = argparse.ArgumentParser()
parser.add_argument("Project", metavar="project", type=str)
args = parser.parse_args()

project = args.Project

load_model = True
fine_tuning = True
training = False
main_folder = "/work3/s203257"
origin = f"{main_folder}/{project}_raw/"
destination = f"{main_folder}/{project}_processed/"
model_path = f"save_model/Myanmar-encode.pt"
save_path = None  # f"save_model/{project}-freezed-new.pt"

logging.info(f"Running project {project} load_model: {load_model} model: {model_path}")

batch_size = 32
input_shape = (192, 192, 3)

dtype = {"frame_id": "str"} if project != "Myanmar" else None
video_frame_as_string = True if project != "Myanmar" else False

train_ids = pd.read_csv(f"{main_folder}/{project}_annot/training_set.csv", dtype=dtype)
val_ids = pd.read_csv(f"{main_folder}/{project}_annot/val_set.csv", dtype=dtype)
test_ids = pd.read_csv(f"{main_folder}/{project}_annot/test_set.csv", dtype=dtype)

dataset_sizes = {"train": len(train_ids), "val": len(val_ids), "test": len(test_ids)}
logging.debug(dataset_sizes)

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
    video_frame_as_string=video_frame_as_string,
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
    video_frame_as_string=video_frame_as_string,
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
    video_frame_as_string=video_frame_as_string,
)

dataloaders = {
    "train": DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4),
    "val": DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=4),
    "test": DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=4),
}

logging.debug(dataloaders)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model_ft = models.resnet34() if load_model else models.resnet34(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, 10)
if load_model:
    model_ft.load_state_dict(torch.load(model_path))

if fine_tuning:
    # freeze the first 6 layers
    ct = 0
    for child in model_ft.children():
        ct += 1
        if ct < 9:
            for param in child.parameters():
                param.requires_grad = False

model_ft = model_ft.to(device)

# Observe that all parameters are being optimized
# optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.0001)
optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.0001)

# Decay LR by a factor of 0.1 every 10 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=5, gamma=0.1)


def train_model(model, optimizer, scheduler, num_epochs=1):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    # best_loss = np.Inf
    best_acc = 0
    epoch_ACCs = []

    train_losses = []
    validation_losses = []

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch + 1, num_epochs))
        print("-" * 10)

        # Each epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0.0

            # Iterate over data.
            for i_batch, sample_batched in tqdm(enumerate(dataloaders[phase])):

                inputs, labels = sample_batched["image"], sample_batched["label"]
                inputs, labels = inputs.type(torch.cuda.FloatTensor), labels.type(torch.cuda.FloatTensor)
                inputs, labels = inputs.to(device), labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                if fine_tuning == True and phase == "train":
                    outputs = model(inputs)
                    # _, preds = torch.max(outputs, 1)
                    num_correct = encode_metric(outputs, labels)
                    # print(preds.shape,labels.shape)
                    loss = loss_fn(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                else:
                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)
                        # _, preds = torch.max(outputs, 1)
                        num_correct = encode_metric(outputs, labels)
                        # print(preds.shape,labels.shape)
                        loss = loss_fn(outputs, labels)

                        # backward + optimize only if in training phase
                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                # torch.sum(preds == labels.data)
                running_corrects += num_correct

            #                 if i_batch % 100 == 99:
            #                     print('[%d, %5d] loss: %.3f' %
            #                           (epoch + 1, i_batch + 1, running_loss / (i_batch*20)))

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            epoch_ACCs.append(epoch_acc.item())

            if phase == "train":
                scheduler.step()
                train_losses.append(epoch_loss)
            if phase == "val":
                validation_losses.append(epoch_loss)

            logging.debug("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    logging.info("Training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))
    # print('Best val loss: {:4f}'.format(best_loss))
    logging.info("Best val Acc: {:4f}".format(best_acc))

    epoch = np.arange(num_epochs)

    plt.figure()
    plt.plot(
        epoch,
        train_losses,
        "r",
        label="Training loss",
    )
    plt.plot(epoch, validation_losses, "b", label="Validation loss")
    plt.legend()
    plt.xlabel("Epoch"), plt.ylabel("NLL")
    plt.show()
    plt.savefig(f"{project}-plot.png")

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, epoch_ACCs


def append_classes_accuracy(total, epoch):
    for key in epoch:
        if key not in total:
            total[key] = {}
            total[key]["num"] = 0
            total[key]["num_correct"] = 0

        total[key]["num"] += epoch[key]["tot"]
        total[key]["num_correct"] += epoch[key]["num_correct"]

    return total


def classes_accuracy(inputs, outputs):
    classes = {}
    for i in range(len(inputs)):
        stringify = "".join(str(int(e)) for e in inputs[i])
        # print(f"{stringify}")
        if stringify not in classes:
            classes[stringify] = {}
            classes[stringify]["tot"] = 0
            classes[stringify]["num_correct"] = 0

        if inputs[i] == outputs[i]:
            classes[stringify]["num_correct"] += 1

        classes[stringify]["tot"] += 1

    return classes


def run_test_on_trained_model(model):
    sigmoid = nn.Sigmoid()
    phase = "test"
    y_pred = []
    running_loss = 0.0
    running_corrects = 0.0
    global_accuracy = {}
    for i_batch, sample_batched in enumerate(dataloaders[phase]):
        inputs, labels = sample_batched["image"], sample_batched["label"]
        inputs, labels = inputs.type(torch.cuda.FloatTensor), labels.type(torch.cuda.FloatTensor)
        inputs, labels = inputs.to(device), labels.to(device)

        outputs = model(inputs)
        outputs_ = sigmoid(outputs).cpu()
        y_pred.append(outputs_.detach().numpy())

        outputs_labels = (sigmoid(outputs) > 0.5).type(torch.FloatTensor)
        accuracy = classes_accuracy(labels.tolist(), outputs_labels.tolist())
        global_accuracy = append_classes_accuracy(global_accuracy, accuracy)

        num_correct = encode_metric(outputs, labels)
        loss = loss_fn(outputs, labels)

        running_loss += loss.item() * inputs.size(0)
        running_corrects += num_correct

        if i_batch % 50 == 49:
            print("[%5d] loss: %.3f" % (i_batch + 1, running_loss / (i_batch * 20)))

    print("-----GLOBAL ACCURACY-------")
    print(global_accuracy)
    for i in global_accuracy:
        i_acc = global_accuracy[i]["num_correct"] / global_accuracy[i]["num"]
        print(f"ID: {i} - Acc: {i_acc} - Tot: {global_accuracy[i]['num']}")

    epoch_loss = running_loss / dataset_sizes[phase]
    epoch_acc = running_corrects.double() / dataset_sizes[phase]

    return epoch_acc, epoch_loss


if training:
    model_ft, epoch_ACCs = train_model(model_ft, optimizer_ft, exp_lr_scheduler, num_epochs=10)
    if save_path:
        torch.save(model_ft.state_dict(), save_path)
logging.debug(epoch_ACCs)

model_ft.eval()

acc, loss = run_test_on_trained_model(model_ft)
logging.info("TEST Loss: {:.4f} Acc: {:.4f}".format(loss, acc))
