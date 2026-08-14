import time
import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam, SGD
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

class FMNIST_Dataset(Dataset):
    def __init__(self, x, y, device):
        x = x.float()/255
        #x = x.float()/(255*10000)
        x = x.view(-1, 28*28)
        self.x, self.y = x, y
        self.device = device

    def __getitem__(self, ix):
        x, y = self.x[ix], self.y[ix]
        device = self.device
        return x.to(device), y.to(device)

    def __len__(self):
        return len(self.x)

def get_data(tr_images, tr_targets, val_images, val_targets, device):
    print(f'Getting FMNIST Data...')
    train = FMNIST_Dataset(tr_images, tr_targets, device)
    train_data = DataLoader(train, batch_size=16, shuffle=True)
    validate = FMNIST_Dataset(val_images, val_targets, device)
    validate_data = DataLoader(validate, batch_size=len(val_images), shuffle=False)
    return train_data, validate_data

def get_model(device):
    print(f'Getting Training Model...')
    class NeuralNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.input_to_hidden_layer = nn.Linear(784, 1000)
            self.batch_norm = nn.BatchNorm1d(1000)
            self.hidden_layer_activation = nn.ReLU()
            self.hidden_to_output_layer = nn.Linear(1000, 10)
        def forward(self, x):
            x = self.input_to_hidden_layer(x)
            x0 = self.batch_norm(x)
            x1 = self.hidden_layer_activation(x0)
            x2 = self.hidden_to_output_layer(x1)
            return x2, x1

    model = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(28*28, 1000),
            nn.BatchNorm1d(1000),
            nn.ReLU(),
            #nn.Dropout(0.25),
            #nn.Linear(1000, 1000),
            #nn.BatchNorm1d(1000),
            #nn.ReLU(),
            nn.Linear(1000, 10)
        ).to(device)
    
    # model = NeuralNet().to(device)
    loss_func = nn.CrossEntropyLoss()
    ## optimizer = SGD(model.parameters(), lr=1e-3)
    optimizer = Adam(model.parameters(), lr=1e-3)
    return model, loss_func, optimizer

def train_batch(x, y, model, optimizer, loss_func):
    model.train()
    prediction = model(x)

    # Apply L1 regularization
    #L1_regularization = 0
    #for param in model.parameters():
    #    L1_regularization += torch.norm(param,1)
    #batch_loss = loss_func(prediction, y) + 0.0001*L1_regularization

    # Apply L2 regularization
    L2_regularization = 0
    for param in model.parameters():
        L2_regularization += torch.norm(param,2)
    batch_loss = loss_func(prediction, y) + 0.01*L2_regularization


    # batch_loss = loss_func(prediction, y)
    batch_loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return batch_loss.item()

@torch.no_grad()
def accuracy(x, y, model):
    model.eval()
    prediction = model(x)
    max_values, argmaxes = prediction.max(-1)
    is_correct = argmaxes == y
    return is_correct.cpu().numpy().tolist()

@torch.no_grad()
def calc_validation_loss(x, y, model, loss_func):
    model.eval()
    prediction = model(x)
    validation_loss = loss_func(prediction, y)
    return validation_loss.item()



data_directory = 'C:/dev/test_scripts/ComputerVisionWithPyTorch/mnist'
print(f'Downloading FMNIST data to {data_directory}')
train_fmnist = datasets.FashionMNIST(data_directory, download=True, train=True)
train_images = train_fmnist.data
train_targets = train_fmnist.targets

validate_fmnist = datasets.FashionMNIST(data_directory, download=True, train=False)
validate_images = validate_fmnist.data
validate_targets = validate_fmnist.targets

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Using processing device: {device}')

train_dataloader, validate_dataloader = get_data(
        train_images,
        train_targets,
        validate_images,
        validate_targets,
        device)

model, loss_func, optimizer = get_model(device)

train_losses, train_accuracies = [], []
validation_losses, validation_accuracies = [], []

start_time = time.time()
num_training_epochs = 10


for epoch in range(num_training_epochs):
    print(f'Running Train Epoch {epoch}')
    train_epoch_losses, train_epoch_accuracies = [], []


    for ix, batch in enumerate(iter(train_dataloader)):
        x, y = batch
        batch_loss = train_batch(x, y, model, optimizer, loss_func)
        train_epoch_losses.append(batch_loss)

    train_epoch_loss = np.array(train_epoch_losses).mean()
    print(f'Epoch {epoch} loss: {train_epoch_loss}')
    train_losses.append(train_epoch_loss)


    for ix, batch in enumerate(iter(train_dataloader)):
        x, y = batch
        is_correct = accuracy(x, y, model)
        train_epoch_accuracies.extend(is_correct)

    train_epoch_accuracy = np.mean(train_epoch_accuracies)
    print(f'Epoch {epoch} accuracy: {train_epoch_accuracy}')
    train_accuracies.append(train_epoch_accuracy)
    

    for ix, batch in enumerate(iter(validate_dataloader)):
        print(f'running validation batch...')
        x, y = batch
        validation_is_correct = accuracy(x, y, model)
        validation_loss = calc_validation_loss(x, y, model, loss_func)

    validation_epoch_accuracy = np.mean(validation_is_correct)
    validation_losses.append(validation_loss)
    validation_accuracies.append(validation_epoch_accuracy)
    print(f'Epoch {epoch} Validation Accuracy: {validation_epoch_accuracy}')

end_time = time.time()
print(f"Training/Validation Execution time: {end_time - start_time} seconds")


epochs = np.arange(num_training_epochs) + 1
#
# Original plain double graph
#
#plt.figure(figsize=(20,5))
#plt.subplot(121)
#plt.title('Loss Value over increasing epochs')
#plt.plot(epochs, train_losses, label='Training Loss')
#plt.legend()
#
#plt.subplot(122)
#plt.title('Accuracy over increasing epochs')
#plt.plot(epochs, train_accuracies, label='Training Accuracy')
#plt.gca().set_yticklabels(['{:0f}%'.format(x*100) \
#        for x in plt.gca().get_yticks()])
#plt.legend()
#plt.show()

#
# new improved double graph
#
plt.subplot(211)
plt.plot(epochs, train_losses, 'bo', label='Training Loss')
plt.plot(epochs, validation_losses, 'r', label='Validation Loss')
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
plt.title('Training and Validation Loss with batch size 32')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid('off')
#plt.show()

plt.subplot(212)
plt.plot(epochs, train_accuracies, 'bo', label='Training Accuracies')
plt.plot(epochs, validation_accuracies, 'r', label='Validation Accuracy')
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
plt.title('Training and Validation Accuracy with batch size 32')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.gca().set_yticklabels(['{:0f}%'.format(x*100) for x in plt.gca().get_yticks()])
plt.legend()
plt.grid('off')
plt.show()







