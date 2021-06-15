# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import cv2
import torchvision.transforms as transforms
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensor
import numpy as np
import torchvision.transforms.functional as TF


def pre_process(x):
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15,15))
  gray = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
  result= cv2.dilate(gray, kernel, iterations=1)
  return result

class Gamified_Sclera_Dataset(Dataset):
  def __init__(self, main_dir, image_shape, transforms):
    super(Gamified_Sclera_Dataset, self).__init__()

    self.main_dir= main_dir
    self.transforms= transforms
    self.image_shape= image_shape

  def __len__(self):
    return len(os.listdir(self.main_dir))

  def __getitem__(self, index):
    dir= os.path.join(self.main_dir, sorted(os.listdir(self.main_dir))[index])
    names_list= sorted(os.listdir(dir))
    image_name, mask_names= names_list[0], names_list[1:]
    image= cv2.resize(cv2.cvtColor(cv2.imread(os.path.join(dir, image_name)), cv2.COLOR_BGR2RGB), self.image_shape)
    masks= np.zeros(self.image_shape+(len(mask_names), ))

    for i in range(len(mask_names)):
      masks[:,:,i]= pre_process(cv2.resize(cv2.imread(os.path.join(dir, mask_names[i])), self.image_shape))
    
    if self.transforms is not None:
      transformed = self.transforms(image=image, mask=masks)
      image = transformed["image"]
      masks = transformed["mask"][0].permute(2, 0, 1)
    return image, masks

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.sigmoid= nn.Sigmoid()
        # Down part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.sigmoid(self.final_conv(x))

class Partial_Categorical_Loss(nn.Module):
    def __init__(self):
        super(Partial_Categorical_Loss, self).__init__()

    def forward(self, y_preds, y_true):
        
        mean= torch.mean(y_true, dim=1)
        mean= torch.unsqueeze(mean.reshape(mean.shape[0], -1), dim=2)
        y_true_bg= torch.zeros_like(mean)
        y_true_bg[mean<0.5]=1

        num_pixels= y_preds.shape[2]*y_preds.shape[3]
        y_preds= y_preds.reshape(y_preds.shape[0], -1)
        y_true= y_true.reshape(y_true.shape[0], y_true.shape[1], -1).permute(0, 2, 1)
        y_true= torch.cat((y_true, y_true_bg), dim=2)

        loss=0
        for i in range(y_true.shape[2]):
            loss+= torch.sum((y_true[:,:,i]*torch.log(y_preds)))

        return -1*loss/num_pixels