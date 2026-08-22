import torch
import numpy as np
import torch.nn as nn
import seaborn as sns
import matplotlib.pyplot as plt
from torchsummary import summary
from torchvision import datasets
from torch.optim import Adam, SGD
import matplotlib.ticker as mticker
from torch.utils.data import Dataset, DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

data_directory = 'C:/dev/test_scripts/ComputerVisionWithPyTorch/mnist'
print(f'Downloading FMNIST data to {data_directory}')

training_fmnist = datasets.FashionMNIST(data_directory, download=True, train=True)
train_images = training_fmnist.data
train_targets = training_fmnist.targets
unique_training_values = train_targets.unique()

validation_fmnist = datasets.FashionMNIST(data_directory, download=True, train=False)
validation_images = validation_fmnist.data
validation_targets = validation_fmnist.targets

print(f'train_images & train_targets shape:\n\tX - {train_images.shape}\n\tY - {train_targets.shape}\n\t\
Y - Unique Values: {unique_training_values}')

print(f'TASK:\n\t {len(unique_training_values)} class Classification')
print(f'UNIQUE CLASSES:\n\t {training_fmnist.classes}')

class FMNIST_Dataset(Dataset):
    def __init__(self, x, y):
        x = x.float()/255
        x = x.view(-1, 1, 28, 28)
        self.x, self.y = x, y

    def __getitem__(self, ix):
        x, y = self.x[ix], self.y[ix]
        return x.to(device), y.to(device)

    def __len__(self):
        return len(self.x)


def get_data():
    training_dataset = FMNIST_Dataset(train_images, train_targets)
    training_dataloader = DataLoader(training_dataset, batch_size=16, shuffle=True)
    validation_dataset = FMNIST_Dataset(validation_images, validation_targets)
    validation_dataloader = DataLoader(validation_dataset, 
            batch_size=len(validation_images), shuffle=False)
    return training_dataloader, validation_dataloader


def get_model():
    model = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3),
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3),
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(3200, 256),
                nn.ReLU(),
                nn.Linear(256, 10)
            ).to(device)
    loss_func = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=1e-3)
    return model, loss_func, optimizer


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
    max_values, argmaxes = prediction.max(-1)
    is_correct = (argmaxes == y)
    return is_correct.cpu().numpy().tolist()

@torch.no_grad()
def validation_loss(x, y, model, loss_func):
    model.eval()
    prediction = model(x)
    val_loss = loss_func(prediction, y)
    return val_loss.item()




model, loss_func, optimizer = get_model()
##summary(model, torch.zeros(1,1,28,28))
summary(model, (1,28,28))



train_dataloader, validation_dataloader = get_data()
model, loss_func, optimizer = get_model()

number_epochs = 5
train_losses, train_accuracies = [], []
val_losses, val_accuracies = [], []


for epoch in range(number_epochs):
    print(f'Starting epoch {epoch}')
    train_epoch_losses, train_epoch_accuracies = [], []

    for ix, batch in enumerate( iter(train_dataloader) ):
        x, y = batch
        batch_loss = train_batch(x, y, model, optimizer, loss_func)
        train_epoch_losses.append(batch_loss)

    train_epoch_loss = np.array(train_epoch_losses).mean()

    for ix, batch in enumerate( iter(train_dataloader) ):
        x, y = batch
        is_correct = accuracy(x, y, model)
        train_epoch_accuracies.extend(is_correct)

    train_epoch_accuracy = np.mean(train_epoch_accuracies)

    train_losses.append(train_epoch_loss)
    train_accuracies.append(train_epoch_accuracy)

    for ix, batch in enumerate(validation_dataloader):
        x, y = batch
        val_is_correct = accuracy(x, y, model)
        val_loss = validation_loss(x, y, model, loss_func)

    val_epoch_accuracy = np.mean(val_is_correct)

    val_losses.append(val_loss)
    val_accuracies.append(val_epoch_accuracy)

#
# graphs for only training losses and accuracies
#
'''
epochs = np.arange(number_epochs) + 1
plt.figure(figsize=(20,5))
plt.subplot(121)
plt.title('Loss value over increasing epochs')
plt.plot(epochs, train_losses, label='Training Loss')
plt.legend()

plt.subplot(122)
plt.title('Accuracy over increasing epochs')
plt.plot(epochs, train_accuracies, label='Training Accuracy')
plt.gca().set_yticklabels( ['{:.0f}%'.format(x*100) \
        for x in plt.gca().get_yticks()])

plt.legend()
plt.show()
'''

#
# graphs for training and validation data losses and accuracies
#
epochs = np.arange(number_epochs) + 1
plt.subplot(211)
plt.plot(epochs, train_losses, "bo", label='Training Loss')
plt.plot(epochs, val_losses, "r", label='Validation Loss')
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
plt.title('Training and Validation Loss values over increasing epochs')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid('off')

plt.subplot(212)
plt.plot(epochs, train_accuracies, 'bo', label='Training Accuracy')
plt.plot(epochs, val_accuracies, 'r', label='Validation Accuracy')
plt.gca().xaxis.set_major_locator(mticker.MultipleLocator(1))
plt.title('Training and Validation Accuracy over increasing epochs')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.gca().set_yticklabels( ['{:.0f}%'.format(x*100) \
        for x in plt.gca().get_yticks()])

plt.legend()
plt.grid('off')
plt.show()

idx = 24300
plt.imshow(train_images[idx], cmap='gray')
plt.title(training_fmnist.classes[train_targets[idx]])
plt.show()

img = train_images[idx].float()/255.
img = img.view(-1, 28*28)
img2 = torch.Tensor(img).view(-1,1,28,28).to(device)
np_output = model(img2).cpu().detach().numpy()
guess = np.exp(np_output)/np.sum(np.exp(np_output))
print (f'Image probablities: {guess}')

preds = []

for px in range(-5, 6):
    img = train_images[idx].float()/255.
    img = img.view(-1, 28*28)
    img2 = np.roll(img, px, axis=1)
    img3 = torch.Tensor(img2).view(-1,1,28,28).to(device)
    np_output = model(img3).cpu().detach().numpy()
    guess = np.exp(np_output)/np.sum(np.exp(np_output))
    #squeezed = np.squeeze(guess, axis=1)
    # print(f'Guess shape: {guess.shape}, Squeezed: {squeezed.shape}')
    preds.append(guess)
    print(f'Guess shape: {guess.shape}')
    print (f'Image probablities: {guess}')
        
np_array = np.squeeze(np.array(preds), axis=1)
print(f'NumPy array shape: {np_array.shape}')

fig, ax = plt.subplots(1, 1, figsize=(12,10))
plt.title('Probability of each class for various translations')
#sns.heatmap(np.array(preds), annot=True, ax=ax, fmt='.2f',
sns.heatmap(np_array, annot=True, ax=ax, fmt='.2f',
        xticklabels=training_fmnist.classes,
        yticklabels=[str(i)+str(' pixels') for i in range(-5,6)], cmap='gray')
plt.show()



