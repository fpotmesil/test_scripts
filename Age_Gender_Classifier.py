import cv2
import torch
import glob, time
import torchvision
import numpy as np
import pandas as pd
import torch.nn as nn
from torch import optim
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models, datasets

SCALED_IMAGE_SIZE = (224,224)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

#--------------------------------------------------------------
#
#
#
#--------------------------------------------------------------
class AgeGenderClassifier(nn.Module):
    def __init__(self):
        super(AgeGenderClassifier, self).__init__()

        self.intermediate = nn.Sequential(
                                nn.Linear(2048,512),
                                nn.ReLU(),
                                nn.Dropout(0.4),
                                nn.Linear(512, 128),
                                nn.ReLU(),
                                nn.Dropout(0.4),
                                nn.Linear(128,64),
                                nn.ReLU())
        self.age_classifier = nn.Sequential(
                                nn.Linear(64,1),
                                nn.Sigmoid())
        self.gender_classifier = nn.Sequential(
                                    nn.Linear(64,1),
                                    nn.Sigmoid())

    def forward(self, x):
        x = self.intermediate(x)
        age = self.age_classifier(x)
        gender = self.gender_classifier(x)
        return gender, age
#--------------------------------------------------------------
#--------------------------------------------------------------
###############################################################
#--------------------------------------------------------------
#
#
#
#--------------------------------------------------------------
def get_model():
    model = models.vgg16(weights='DEFAULT')

    for param in model.parameters():
        param.requires_grad = False

    model.avgpool = nn.Sequential(
                        nn.Conv2d(512, 512, kernel_size=3),
                        nn.MaxPool2d(2),
                        nn.ReLU(),
                        nn.Flatten())

    model.classifier = AgeGenderClassifier()
    gender_loss = nn.BCELoss()
    age_loss = nn.L1Loss()
    loss_functions = age_loss, gender_loss
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    return model.to(device), loss_functions, optimizer
#--------------------------------------------------------------
#--------------------------------------------------------------

#--------------------------------------------------------------
#
#
#
#--------------------------------------------------------------
class AgeGenderDataset(Dataset):
    def __init__(self, df, tfms=None):
        self.df = df
        self.normalize = transforms.Normalize(
                mean = [0.485, 0.456, 0.406],
                std = [0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, ix):
        f = self.df.iloc[ix].squeeze()
        file = root_data_dir + f.file
        gender = f.gender == 'Female'
        age = f.age
        print( f'opening image path {file}')
        img = cv2.imread(file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img, age, gender

    def preprocess_image(self, img):
        print(f'original image shape: {img.shape}')
        img = cv2.resize(img, SCALED_IMAGE_SIZE)
        
        img = torch.tensor(img)
        print(f'image tensor shape: {img.shape}')

        img = img.permute(2,0,1)
        print(f'image tensor permuted shape: {img.shape}')


        img = self.normalize(img/255.0)

        # Convert to tensor (values in [0,1], shape: C x H x W)
        # to_tensor = transforms.ToTensor()
        # img_tensor = to_tensor(img)  # torch.Size([3, H, W])
        
        # Add batch dimension (1 x C x H x W)
        img = img.unsqueeze(0)
        print(f'image tensor permuted unsqueezed shape: {img.shape}')

        return img
        # instead of less clear way to add a batch dimension:
        ## return im[None]

    def collate_func(self, batch):
        imgs, ages, genders = [], [], []

        for img, age, gender in batch:
            img = self.preprocess_image(img)
            imgs.append(img)
            ages.append(float(int(age)/80))
            genders.append(float(gender))

        ages, genders = \
            [torch.tensor(x).to(device).float() for x in [ages, genders]]
            
        imgs = torch.cat(imgs).to(device)

        return imgs, ages, genders    
#--------------------------------------------------------------
#--------------------------------------------------------------
#
#--------------------------------------------------------------
#
#
#
#--------------------------------------------------------------
def train_batch(data, model, optimizer, criteria):
    model.train()
    imgs, age, gender = data
    optimizer.zero_grad()
    pred_gender, pred_age = model(imgs)
    age_loss_func, gender_loss_func = criteria
    age_loss = age_loss_func(pred_age.squeeze(), age)
    gender_loss = gender_loss_func(pred_gender.squeeze(), gender)
    total_loss = age_loss + gender_loss
    total_loss.backward()
    optimizer.step()
    return total_loss
#--------------------------------------------------------------
#--------------------------------------------------------------
#
#--------------------------------------------------------------
#
#
#
#--------------------------------------------------------------
def validate_batch(data, model, criteria):
    model.eval()
    imgs, age, gender = data
    
    with torch.no_grad():
        pred_gender, pred_age = model(imgs)
    
    age_loss_func, gender_loss_func = criteria
    age_loss = age_loss_func(pred_age.squeeze(), age)
    gender_loss = gender_loss_func(pred_gender.squeeze(), gender)
    total_loss = age_loss + gender_loss

    pred_gender = (pred_gender > 0.5).squeeze()
    gender_accuracy = (pred_gender == gender).float().sum()
    age_mae = torch.abs(age-pred_age).float().sum()
    return total_loss, age_mae, gender_accuracy



#--------------------------------------------------------------
#--------------------------------------------------------------

root_data_dir = 'c:/dev/test_scripts/ComputerVisionWithPyTorch/FairFace/'
train_df = pd.read_csv(root_data_dir + 'fairface-labels-train.csv')
validation_df = pd.read_csv(root_data_dir + 'fairface-labels-val.csv')

print(f'Fair Face Training dataframe info:')
print(train_df.info())

print(f'Fair Face Training dataframe first 10 entries:')
print(train_df.head(10))

train_dataset = AgeGenderDataset(train_df)
validation_dataset = AgeGenderDataset(validation_df)

train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True,
        drop_last=True, collate_fn = train_dataset.collate_func)

validate_dataloader = DataLoader(validation_dataset, batch_size=32,
        collate_fn = validation_dataset.collate_func)

a,b,c = next( iter(train_dataloader) )
print(f'a.shape: {a.shape}, b.shape: {b.shape}, c.shape: {c.shape}')

model, criteria, optimizer = get_model()

validate_age_maes = []
validate_gender_accuracies = []
validate_losses = []
train_losses = []

n_epochs = 5
best_test_loss = 1000
start = time.time()

for epoch in range(n_epochs):
    print(f'Starting epoch {epoch+1}')
    epoch_train_loss, epoch_test_loss = 0, 0
    validate_age_mae, validate_gender_accuracy, ctr = 0, 0, 0
    _n = len(train_dataloader)

    for ix, data in enumerate(train_dataloader):
        loss = train_batch(data, model, optimizer, criteria)
        epoch_train_loss += loss.item()

    for ix, data in enumerate(validate_dataloader):
        loss, age_mae, gender_accuracy = validate_batch(data, model, criteria)
        epoch_test_loss += loss.item()
        validate_age_mae += age_mae
        valiate_gender_accuracy += gender_accuracy
        ctr += len(data[0])

    validate_age_mae /= ctr
    validate_gender_accuracy /= ctr
    epoch_train_loss /= len(train_dataloader)
    epoch_test_loss /= len(validate_dataloader)
    
    elapsed = time.time()-start
    best_test_loss = min(best_test_loss, epoch_test_loss)

    print('{}/{} ({:.2f}s - {:.2f}s remaining)'.format(\
            epoch+1, n_epochs, time.time()-start, \
            (n_epochs-epoch)*(elapsed/(epoch+1))))

    info = f'''Epoch: {epoch+1:03d}
            \tTrainLoss: {epoch_train_loss:.3f}
            \tTest Loss: {epoch_test_loss:.3f}
            \tBest Test Loss: {best_test_loss:.4f}'''

    info += f'\nGender Accuracy:\
                {validate_gender_accuracy*100:.2f}%\tAge MAE: {validate_age_mae:.2f}\n'

    print(info)

    validate_gender_accuracies.append(validate_gender_accuracy)
    validate_age_maes.append(validate_age_mae)

epochs = np.arange(1,(n_epochs+1))
fig, ax = plt.subplots(1, 2, figsize=(10,5))
ax = ax.flat
ax[0].plot(epochs, validate_gender_accuracies, 'bo')
ax[1].plot(epochs, validate_age_maes, 'r')
ax[0].set_xlabel('Epochs')
ax[1].set_xlabel('Epochs')
ax[0].set_ylabel('Accuracy')
ax[1].set_ylabel('MAE')
ax[0].set_title('Validation Gender Accuracy')
ax[1].set_title('Validation Age Mean-Absolute-Error')
plt.show()









