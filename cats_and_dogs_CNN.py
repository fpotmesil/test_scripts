import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torchvision import transforms, models, datasets
from PIL import Image
from torch import optim
import cv2
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from torch.utils.data import DataLoader, Dataset
from random import shuffle, seed
from torchsummary import summary

device = 'cuda' if torch.cuda.is_available() else 'cpu'

test_data_dir = 'C:/dev/test_scripts/ComputerVisionWithPyTorch/cats-and-dogs/test_set' 
training_data_dir = 'C:/dev/test_scripts/ComputerVisionWithPyTorch/cats-and-dogs/training_set'

class cats_and_dogs_dataset(Dataset):
    def __init__(self, directory):
        cats = glob(directory + '/cats/*.jpg')
        dogs = glob(directory + '/dogs/*.jpg')
        self.fpaths = cats + dogs
        shuffle(self.fpaths)
        self.targets = [fpath.split('/')[-1].startswith('dog') \
                for fpath in self.fpaths]

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, ix):
        f = self.fpaths[ix]
        target = self.targets[ix]
        im  = (cv2.imread(f)[:,:,::-1])
        im = cv2.resize(im, (224,224))
        return torch.tensor(im/255).permute(2,0,1).to(device).float(),\
                torch.tensor([target]).float().to(device)


data = cats_and_dogs_dataset(training_data_dir)
im, label = data[200]
plt.imshow(im.permute(1,2,0).cpu())
print(label)
plt.show()

def convolution_layer(in_channels, out_channels, kernel_size, stride=1):
    return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                kernel_size=kernel_size, stride=stride),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.MaxPool2d(2))

def get_model():
    model = nn.Sequential(
            convolution_layer(3, 64, 3),
            convolution_layer(64, 512, 3),
            convolution_layer(512, 512, 3),
            convolution_layer(512, 512, 3),
            convolution_layer(512, 512, 3),
            convolution_layer(512, 512, 3),
            nn.Flatten(),
            nn.Linear(512,1),
            nn.Sigmoid()).to(device)
    loss_func = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-3)
    return model, loss_func, optimizer

model, loss_func, optimizer = get_model()
##summary(model, torch.zeros(1,3,224,224))
summary(model, (3,224,224))

def get_data():
    train = cats_and_dogs_dataset(training_data_dir)
    train_dataloader = DataLoader(train, batch_size=32, 
            shuffle=True, drop_last=True)

    validate = cats_and_dogs_dataset(test_data_dir)
    validate_dataloader = DataLoader(validate, batch_size=32,
            shuffle=True, drop_last=True)
    
    return train_dataloader, validate_dataloader

def train_batch(x, y, model, optimizer, loss_func):
    model.train()
    prediction = model(x)
    batch_loss = loss_func(prediction, y)
    optimizer.step()
    optimizer.zero_grad()
    return batch_loss.item()

@torch.no_grad()
def accuracy(x, y, model):
    prediction = model(x)
    is_correct = (prediction > 0.5) == y
    return is_correct.cpu().numpy().tolist()

@torch.no_grad()
def validation_loss(x, y, model):
    prediction = model(x)
    validation_loss = loss_func(prediction,y)
    return validation_loss.item()

train_dataloader, validation_dataloader = get_data()
model, loss_func, optimizer = get_model()

epochs = 5
train_losses, train_accuracies = [], []
validation_losses, validation_accuracies = [], []

for epoch in range(epochs):
    print(f'Running training and validation for epoch {epoch}')
    train_epoch_losses, train_epoch_accuracies = [], []
    validate_epoch_accuracies = []

    for ix, batch in enumerate( iter(train_dataloader) ):
        x, y = batch
        batch_loss = train_batch(x, y, model, optimizer, loss_func)
        train_epoch_losses.append(batch_loss)
    train_epoch_loss = np.array(train_epoch_losses).mean()
    print( f'Training loss for epoch {epoch}: {train_epoch_loss}')

    for ix, batch in enumerate( iter(train_dataloader) ):
        x, y = batch
        is_correct = accuracy(x, y, model)
        train_epoch_accuracies.extend(is_correct)
    train_epoch_accuracy = np.mean(train_epoch_accuracies)
    print(f'Training accuracy for epoch {epoch}: {train_epoch_accuracy}')

    for ix, batch in enumerate( iter(validation_dataloader) ):
        x, y = batch
        val_is_correct = accuracy(x, y, model)
        val_epoch_accuracies.extend(val_is_correct)
    validate_epoch_accuracy = np.mean(val_epoch_accuracies)
    print(f'Validation accuracy for epoch {epoch}: {validate_epoch_accuracy}')

    train_losses.append(train_epoch_loss)
    train_accuracies.append(train_epoch_accuracy)
    validation_accuracies.append(validate_epoch_accuracy)


plt.plot(epochs, train_accuracies, 'bo', label='Training Accuracy')
plt.plot(epochs, validation_accuracies, 'r', label='Validation Accuracy')
plt.gca().xaxis.set_major_locator( mticker.MultipleLocator(1) )
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()])
plt.legend()
plt.grid('off')
plt.show()







