import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torchvision import transforms, models, datasets
from torchsummary import summary
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from PIL import Image
from torch import optim
import cv2
from glob import glob
import numpy as np
import pandas as pd
from random import shuffle, seed
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'

resnet18_model = models.resnet18(weights='DEFAULT').to(device)

summary(resnet18_model, (3,224,224))
print(resnet18_model)

test_data_dir = 'C:/dev/test_scripts/ComputerVisionWithPyTorch/cats-and-dogs/test_set' 
training_data_dir = 'C:/dev/test_scripts/ComputerVisionWithPyTorch/cats-and-dogs/training_set'

class CatsAndDogs(Dataset):
    def __init__(self, directory):
        cats = glob(directory + '/cats/*.jpg')
        dogs = glob(directory + '/dogs/*.jpg')
        self.fpaths = cats[:500] + dogs[:500]
        self.normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
        seed(10)
        shuffle(self.fpaths)
        self.targets = [fpath.split('/')[-1].startswith('dog') for fpath in self.fpaths]

    def __len__(self):
        return len(self.fpaths)

    def __getitem__(self, ix):
        f = self.fpaths[ix]
        target = self.targets[ix]
        img = (cv2.imread(f)[:,:,::-1])
        img = cv2.resize(img, (224, 224))
        img = torch.tensor(img/255)
        img = img.permute(2, 0, 1)
        img = self.normalize(img)

        return (img.float().to(device),
                torch.tensor([target]).float().to(device))

data = CatsAndDogs(training_data_dir)
im, label = data[300]
plt.imshow(im.permute(1,2,0).cpu())
print(label)
plt.show()

# Mean and std used during normalization
mean = torch.tensor([0.485, 0.456, 0.406])
std = torch.tensor([0.229, 0.224, 0.225])

# Denormalize: x * std + mean
img_denorm = im.cpu() * std[:, None, None] + mean[:, None, None]

# Clip to [0, 1] to avoid imshow warnings
img_denorm = torch.clamp(img_denorm, 0, 1)

# Convert to (H, W, C) for imshow
plt.imshow(img_denorm.permute(1, 2, 0).cpu().numpy())
plt.show()


img, label = data[200]
print(label)
plt.imshow(img.permute(1,2,0).cpu())
plt.show()

# Denormalize: x * std + mean
img_denorm = img.cpu() * std[:, None, None] + mean[:, None, None]

# Clip to [0, 1] to avoid imshow warnings
img_denorm = torch.clamp(img_denorm, 0, 1)

# Convert to (H, W, C) for imshow
plt.imshow(img_denorm.permute(1, 2, 0).cpu().numpy())
plt.show()

def get_model():
    model = models.resnet18(weights='DEFAULT').to(device)

    for param in model.parameters():
        param.requires_grad = False

    model.avgpool = nn.AdaptiveAvgPool2d(output_size=(1,1))
    model.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512,128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128,1),
            nn.Sigmoid())
    loss_func = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    return model.to(device), loss_func, optimizer

model, loss_func, optimizer = get_model()
summary(model, (3,224,224))
print(model)
    
def train_batch(x, y, model, optimizer, loss_func):
    model.train()
    prediction = model(x)
    batch_loss = loss_func(prediction, y)
    batch_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return batch_loss.item()

@torch.no_grad()
def accuracy(x, y, model):
    model.eval()
    prediction = model(x)
    is_correct = (prediction > 0.5) == y
    return is_correct.cpu().numpy().tolist()

def get_data():
    train = CatsAndDogs(training_data_dir)
    validate = CatsAndDogs(test_data_dir)

    train_dataloader = DataLoader(train, batch_size=32, 
            shuffle=True, drop_last=True)

    validate_dataloader = DataLoader(validate, batch_size=32, 
            shuffle=True, drop_last=True)

    return train_dataloader, validate_dataloader

train_data, validate_data = get_data()
model, loss_func, optimizer = get_model()

train_losses, train_accuracies = [], []
validate_accuracies = []

for epoch in range(5):
    print(f'Running Epoch {epoch+1}')
    train_epoch_losses, train_epoch_accuracies = [], []
    validate_epoch_accuracies = []

    for ix, batch in enumerate( iter(train_data) ):
        x, y = batch
        batch_loss = train_batch(x, y, model, optimizer, loss_func)
        train_epoch_losses.append(batch_loss)
    train_epoch_loss = np.array(train_epoch_losses).mean()

    for ix, batch in enumerate( iter(train_data) ):
        x, y = batch
        is_correct = accuracy(x, y, model)
        train_epoch_accuracies.extend(is_correct)
    train_epoch_accuracy = np.mean(train_epoch_accuracies)

    for ix, batch in enumerate( iter(validate_data) ):
        x, y = batch
        val_is_correct = accuracy(x, y, model)
        validate_epoch_accuracies.extend(val_is_correct)
    validate_epoch_accuracy = np.mean(validate_epoch_accuracies)

    train_losses.append(train_epoch_loss)
    train_accuracies.append(train_epoch_accuracy)
    validate_accuracies.append(validate_epoch_accuracy)

epochs = np.arange(5) + 1
plt.plot(epochs, train_accuracies, 'bo', label='Training Accuracy')
plt.plot(epochs, validate_accuracies, 'r', label='Validation Accuracy')
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
plt.title('Training and Validation Accuracy with VGG16')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.ylim(0.95,1)
plt.gca().set_yticklabels(['{:.0f}%'.format(x*100) for x in plt.gca().get_yticks()])
plt.legend()
plt.grid('off')
plt.show()





