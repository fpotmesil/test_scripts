import torch
import numpy as np
from torch import nn
import matplotlib.pyplot as plt
from torchvision import datasets
from torch.optim import SGD, Adam
from torch.utils.data import TensorDataset, Dataset, DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'


#
# tensor size is (2,1,4,4)
#
# this can be thought of kinda like:
# having two sample images
# each image has one channel - grayscale
# each image is only 4px x 4px
#

X_train = torch.tensor([
    [
        [
            [1,2,3,4],
            [2,3,4,5],
            [5,6,7,8],
            [1,3,4,5]
        ],
    ],
    [
        [
            [-1,2,3,-4],
            [2,-3,4,5],
            [-5,6,-7,8],
            [-1,-3,-4,-5]
        ]
    ]
    ]).to(device).float()

## X_train = torch.tensor([[[[1,2,3,4],[2,3,4,5],[5,6,7,8],[1,3,4,5]]],[[[-1,2,3,-4],[2,-3,4,5],[-5,6,-7,8],[-1,-3,-4,-5]]]]).to(device).float()

print( f'X_train shape: {X_train.shape}')
zeros = torch.zeros_like(X_train)
print(f'Zeros tensor shape: {zeros.shape}')
print(zeros)

#
# scale the X train data by dividing by the greatest value
# scaling forces training data to be constrained by -1 and +1
# 
X_train /= 8 
print(f'After scaling X_train:\n{X_train}')

y_train = torch.tensor([0,1]).to(device).float()

def get_model():
    model = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=3),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(1, 1),
            nn.Sigmoid(),
            ).to(device)

    loss_func = nn.BCELoss()
    optimizer = Adam(model.parameters(), lr=1e-3)

    return model, loss_func, optimizer

def train_batch(x, y, model, optimizer, loss_func):
    model.train()
    prediction = model(x)
    batch_loss = loss_func(prediction.squeeze(0), y)
    batch_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return batch_loss.item()


from torchsummary import summary
model, loss_func, optimizer = get_model()
summary(model, (1,4,4))
# summary(model, X_train)

train_dataloader = DataLoader(
        TensorDataset(X_train, y_train))

for epoch in range(2000):
    for ix, batch in enumerate(
            iter(train_dataloader)):
        x, y = batch
        batch_loss = train_batch(x, y, model, optimizer, loss_func)

print(f'After training, forward pass on\n{X_train[:1]}\nis:\n{model(X_train[:1])}')


