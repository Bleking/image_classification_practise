import os
import matplotlib.pyplot as plt
from PIL import Image

import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split

import timm

import pickle
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

dir_main = "./Garbage_classification/"
phase = "raw"
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

np.random.seed(724)

directory = os.path.join(dir_main, phase)  # "/content/Garbage_classification/raw/"
classes = glob(f"{directory}/*")
class_info = {idx : os.path.basename(cls) for idx, cls in enumerate(classes)}

img_files = glob(f"{directory}/*/*.jpg")
dataset = np.array([[img_file, img_file.split("/")[-2]] for img_file in img_files])

X = dataset[:, 0]
Y = dataset[:, 1]
x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, random_state=724)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, train_size=0.8, random_state=724)

def create_symlink(x_target, name):
  for x in x_target:
    src = os.path.abspath(x)
    dst = src.replace("raw", name)
    os.makedirs(os.path.dirname(dst), exist_ok=True)

    if not os.path.exists(dst):
      os.symlink(src, dst)


create_symlink(x_train, "train")
create_symlink(x_val, "val")
create_symlink(x_test, "test")

batch_size = 64

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])
test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

train_dataset = ImageFolder(root=os.path.join(dir_main, "train"), transform=train_transform)
val_dataset = ImageFolder(root=os.path.join(dir_main, "val"), transform=test_transform)
test_dataset = ImageFolder(root=os.path.join(dir_main, "test"), transform=test_transform)

train_dataloader = DataLoader(train_dataset, shuffle=True, num_workers=0, batch_size=batch_size)
val_dataloader = DataLoader(val_dataset, shuffle=False, num_workers=0, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, shuffle=False, num_workers=0, batch_size=batch_size)

model = timm.create_model('efficientnet_b0', pretrained=True)
model.default_cfg

model.reset_classifier(len(class_names), 'max')
num_in_features = model.get_classifier().in_features

model.fc = nn.Sequential(
    nn.BatchNorm1d(num_in_features),
    nn.Linear(in_features=num_in_features, out_features=512, bias=False),
    nn.ReLU(),
    nn.BatchNorm1d(512),
    nn.Dropout(0.4),
    nn.Linear(in_features=512, out_features=len(class_names), bias=False)
)

if torch.cuda.is_available():
  device = torch.device('cuda')
else:
  device = torch.device('cpu')


class EfficientNet(nn.Module):
  def __init__(self):
    super().__init__()
    self.model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=len(class_names), global_pool='avg')

  def forward(self, x):
    x = self.model(x)
    x = torch.sigmoid(x)

    return x


model = EfficientNet().to(device)

criterion = nn.CrossEntropy()
optimizer = optim.Adam(model.parameters(), 5.5e-5)

n_epochs = 20
val_acc_best = 0
