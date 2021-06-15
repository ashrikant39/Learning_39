# -*- coding: utf-8 -*-
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
import torch.optim as optim
import cv2
import torchvision.transforms as transforms
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np
import matplotlib.pyplot as plt
from torchsummary import summary
import math
import pkbar
from tqdm import tqdm
from utils import UNET, Gamified_Sclera_Dataset, Partial_Categorical_Loss

loss= Partial_Categorical_Loss()
a= abs(torch.randn((10, 1, 512, 512)))
b= abs(torch.randn((10, 16, 512, 512)))
c= loss(a,b)

path= os.getcwd()
image_shape=(512, 512)
batch_size=4
num_epochs=20
learning_rate=1e-4
device= "cuda" if torch.cuda.is_available() else "cpu"
main_dir= os.path.join(path, 'Dataset2')
dir='weights'
print(len(os.listdir(main_dir)))

check_pt_file=os.path.join(dir, 'UNET_'+str(learning_rate)+'.pth.tar')
print(check_pt_file)

dataset= Gamified_Sclera_Dataset(main_dir, image_shape, transforms=ToTensorV2())
print(f"Dataset size= {dataset.__len__()} images")
dataloader= DataLoader(dataset, batch_size=4)
generator=torch.Generator().manual_seed(42)

train_dataset, _dataset= random_split(dataset, [int(0.8*dataset.__len__()), int(0.2*dataset.__len__())], generator=generator)
val_dataset, test_dataset= random_split(_dataset, [int(0.5*_dataset.__len__()), int(0.5*_dataset.__len__())], generator=generator)

train_loader= DataLoader(train_dataset, batch_size=batch_size)
val_loader= DataLoader(val_dataset, batch_size=batch_size)
test_loader= DataLoader(test_dataset, batch_size=batch_size)

model= UNET().to(device=device)
Loss= Partial_Categorical_Loss()
optimizer= optim.Adam(model.parameters(), lr=learning_rate)

def save_checkpoint(model, optimizer, file_name):

  checkpoint= {'state_dict': model.state_dict(),
             'optimizer_dict': optimizer.state_dict()}
  torch.save(checkpoint,file_name)

def load_checkpoint(model, optimizer, file_name):
  check_pt= torch.load(file_name, map_location= torch.device(device))
  model.load_state_dict(check_pt['state_dict'])
  optimizer.load_state_dict(check_pt['optimizer_dict'])

  return model, optimizer

def dice_coef(scores, targets):
  
  smooth = 1e-3
  scores = torch.flatten(scores.permute(0, 2, 3, 1), start_dim=0, end_dim=2).permute(1, 0)
  targets = torch.flatten(targets.permute(0, 2, 3, 1), start_dim=0, end_dim=2).permute(1, 0)
  dice= []
  for i in range(targets.shape[0]):
    intersection = (scores*targets[i]).sum()
    union = (scores+targets[i]).sum()
    dice.append(((2*intersection + smooth)/(union+smooth)).item())
  return np.mean(dice)


for i in range(batch_size):
    fig = plt.figure(figsize=(10, 4))

    ax1 = fig.add_subplot(1, 4, 1)
    ax1.set_title('Actual frame')
    ax1.imshow(x[i].permute(1, 2, 0).cpu().numpy())

    ax2 = fig.add_subplot(1, 4, 2)
    ax2.set_title('Ground truth labels')
    ax2.imshow(y[i][0].cpu().numpy(), 'gray')

    ax3 = fig.add_subplot(1, 4, 3)
    ax3.set_title('Predictions')
    ax3.imshow(yhat.detach().cpu().numpy()[i][0], 'gray')

    ax4 = fig.add_subplot(1, 4, 4)
    ax4.set_title('Predictions')
    ax4.imshow(logits[i][0].cpu().numpy(), 'gray')

    plt.show()
#
# train_per_epoch = len(train_loader)
# val_per_epoch = len(val_loader)
# min_loss = math.inf
#
# for epoch in range(num_epochs):
#   train_losses = []
#   train_dcs = []
#   kbar_train = pkbar.Kbar(target = train_per_epoch, epoch = epoch, num_epochs = num_epochs)
#   model.train()
#   for batch_idx, (data, targets) in enumerate(train_loader):
#     data = data.to(device=device)
#     targets = targets.to(device=device)
#
#     scores = model(data)
#     train_loss = Loss(scores, targets)
#     train_losses.append(train_loss.item())
#
#     optimizer.zero_grad()
#     torch.autograd.set_detect_anomaly(True)
#     train_loss.backward()
#     optimizer.step()
#
#
#
#     train_dc = dice_coef(scores, targets)
#     train_dcs.append(train_dc.item())
#     kbar_train.update(batch_idx, values=[("loss", train_loss.item()), ("dice_score", train_dc.item())])
#
#   mean_train_loss = np.mean(train_losses)
#   mean_train_dc = np.mean(train_dcs)
#
#   kbar_train.update(train_per_epoch, values=[("loss", mean_train_loss), ("dice_score", mean_train_dc)])#For each epoch
#
#   val_losses = []
#   val_dcs = []
#   kbar_val = pkbar.Kbar(target = val_per_epoch, epoch = epoch, num_epochs = num_epochs)
#   with torch.no_grad():
#     model.eval()
#     for batch_idx, (data, targets) in enumerate(val_loader):
#       data = data.to(device=device)
#       targets = targets.to(device=device)
#
#       scores = model(data)
#       val_loss = Loss(scores, targets)
#       val_losses.append(val_loss.item())
#
#       val_dc = dice_coef(scores, targets)
#       val_dcs.append(val_dc.item())
#       kbar_val.update(batch_idx, values=[("val_loss", val_loss.item()), ("val_dice_score", val_dc.item())])
#
#     mean_val_loss = np.mean(val_losses)
#     mean_val_dc = np.mean(val_dcs)
#     kbar_val.update(val_per_epoch, values=[("val_loss", mean_val_loss), ("val_dice_score", mean_val_dc)])
#
#     if np.mean(val_losses) < min_loss:
#       min_loss = np.mean(val_losses)
#       print('\nImproved validation loss: {:.4f}'.format(min_loss))
#       save_checkpoint(model, optimizer, check_pt_file)
#       print('Model saved to {}\n'.format(check_pt_file))
#
# unet, _= load_checkpoint(model, optimizer, check_pt_file)
# # torch.save(unet, model_path)
#
# x, y= next(iter(test_loader))
# x= x.to(device=device)
# x= x.to(device=device)
# unet.eval()
# yhat= unet(x)
#
# logits= torch.zeros_like(yhat)
# logits[yhat>0.9]=1
#
# for i in range(batch_size):
#
#     fig = plt.figure(figsize=(10,4))
#
#     ax1 = fig.add_subplot(1,4,1)
#     ax1.set_title('Actual frame')
#     ax1.imshow(x[i].permute(1,2,0).cpu().numpy())
#
#
#     ax2 = fig.add_subplot(1,4,2)
#     ax2.set_title('Ground truth labels')
#     ax2.imshow(y[i][0].cpu().numpy(), 'gray')
#
#     ax3 = fig.add_subplot(1,4,3)
#     ax3.set_title('Predictions')
#     ax3.imshow(yhat.detach().cpu().numpy()[i][0], 'gray')
#
#     ax4 = fig.add_subplot(1,4,4)
#     ax4.set_title('Predictions')
#     ax4.imshow(logits[i][0].cpu().numpy(), 'gray')
#
#     plt.show()