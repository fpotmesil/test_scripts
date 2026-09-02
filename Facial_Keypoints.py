import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models, datasets
from torchsummary import summary
import numpy as np
import pandas as pd
import os, glob, cv2
from torch.utils.data import TensorDataset, DataLoader, Dataset
from copy import deepcopy
from mpl_toolkits.mplot3d import Axes3D
from sklearn import cluster
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

device = 'cuda' if torch.cuda.is_available() else 'cpu'

root_data_dir = 'C:/dev/test_scripts/ComputerVisionWithPyTorch/P1_Facial_Keypoints'
training_data = root_data_dir + '/data/training/'
all_img_paths = glob.glob( os.path.join(training_data, '*.jpg') )
training_csv = root_data_dir + '/data/training_frames_keypoints.csv'
data_csv_df = pd.read_csv(training_csv)

print( f'Read {training_csv}, info:\n')
print(data_csv_df.info())
print( f'First rows from {training_csv}:\n')
print(data_csv_df.head(10))
print( f'DataFrame from {training_csv} len:  {len(data_csv_df)}')

class FacesData(Dataset):
    def __init__(self, df):
        super(FacesData).__init__()
        self.df = df
        self.normalize = transforms.Normalize(
                mean = [0.485, 0.456, 0.406],
                std = [0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, ix):
        img_path = training_data + self.df.iloc[ix,0]
        print(f'processing image: {img_path}')
        img = cv2.imread(img_path)/255.
        kp = deepcopy(self.df.iloc[ix,1:].tolist())
        kp_x = (np.array(kp[0::2])/img.shape[1]).tolist()
        kp_y = (np.array(kp[1::2])/img.shape[0]).tolist()
        kp2 = kp_x + kp_y
        kp2 = torch.tensor(kp2)
        img = self.preprocess_input(img)
        return img, kp2

    def preprocess_input(self, img):
        img = cv2.resize(img, (224, 224))
        img = torch.tensor(img).permute(2,0,1)
        img = self.normalize(img).float()
        return img.to(device)

    def load_img(self, ix):
        img_path = training_data + self.df.iloc[ix,0]
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255.
        img = cv2.resize(img, (224, 224))
        return img

def get_model():
    model = models.vgg16(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.avgpool = nn.Sequential(
                        nn.Conv2d(512, 512, 3),
                        nn.MaxPool2d(2),
                        nn.Flatten())
    
    model.classifier = nn.Sequential(
                            nn.Linear(2048,512),
                            nn.ReLU(),
                            nn.Dropout(0.5),
                            nn.Linear(512,136),
                            nn.Sigmoid())
    
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    return model.to(device), criterion, optimizer

def train_batch(img, kps, model, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    _kps = model(img.to(device))
    loss = criterion(_kps, kps.to(device))
    loss.backward()
    optimizer.step()
    return loss

def validate_batch(img, kps, model, criterion):
    model.eval()
    _kps = model(img.to(device))
    loss = criterion(_kps, kps.to(device))
    return _kps, loss



train, test = train_test_split(data_csv_df, test_size=0.2, random_state=101)
train_dataset = FacesData(train.reset_index(drop=True))
test_dataset = FacesData(test.reset_index(drop=True))

train_loader = DataLoader(train_dataset, batch_size=32)
test_loader = DataLoader(test_dataset, batch_size=32)

model, criterion, optimizer = get_model()

train_loss, test_loss = [], []
n_epochs = 50

for epoch in range(n_epochs):
    print(f'Epoch {epoch+1}...')
    epoch_train_loss, epoch_test_loss = 0, 0

    for ix, (img,kps) in enumerate(train_loader):
        loss = train_batch(img, kps, model, optimizer, criterion)
        epoch_train_loss += loss.item()
    epoch_train_loss /= (ix+1)

    for ix, (img, kps) in enumerate(test_loader):
        ps, loss = validate_batch(img, kps, model, criterion)
        epoch_test_loss += loss.item()
    epoch_test_loss /= (ix+1)

    train_loss.append(epoch_train_loss)
    test_loss.append(epoch_test_loss)

epochs = np.arange(n_epochs)+1
plt.plot(epochs, train_loss, 'bo', label='Training Loss')
plt.plot(epochs, test_loss, 'r', label='Test Loss')
plt.title('Training and Test Losses over increasing epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.grid('off')
plt.show()

for ix in range(5):
    plt.figure(figsize=(10,10))
    plt.subplot(221)
    plt.title('Original Image')
    img = test_dataset.load(ix)
    plt.imshow(img)
    plt.grid(False)

    plt.subplot(222)
    plt.title('Image with Facial Keypoints')
    x, _ = test_dataset[ix]
    plt.imshow(img)
    kp = model(x[None]).flatten().detach().cpu()
    plt.scatter(kp[:68]*224, kp[68:]*224, c='r')
    plt.grid(False)
    plt.show()


print('Okey-dokey, that is all now!')



