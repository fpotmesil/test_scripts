import time
import torch
import torch.nn as nn
from torch.optim import SGD
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

x = [[1,2], [3,4], [5,6], [7,8]]
y = [[3], [7], [11], [15]]

X = torch.tensor(x).float()
Y = torch.tensor(y).float()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

X = X.to(device)
Y = Y.to(device)

class MyDataSet(Dataset):
    def __init__(self, x, y):
        if isinstance(x, list):
            self.x = torch.tensor(x).float().to(device)
        elif isinstance(x, torch.Tensor):
            self.x = x.detach().clone().float().to(device)
        else:
            print(f"Invalid Input!  Expected a list, got: {type(x).__name__}")

        if isinstance(y, list):
            self.y = torch.tensor(y).float().to(device)
        elif isinstance(y, torch.Tensor):
            self.y = y.detach().clone().float().to(device)
        else:
            print(f"Invalid Input!  Expected a list, got: {type(y).__name__}")

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class MyNeuralNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_to_hidden_layer = nn.Linear(2,8)
        self.hidden_layer_activation = nn.ReLU()
        self.hidden_to_output_layer = nn.Linear(8,1)
        self.loss_func = nn.MSELoss()
        self.optimizer = SGD(self.parameters(), lr = 0.001)

    def forward(self, x):
        x = self.input_to_hidden_layer(x)
        x = self.hidden_layer_activation(x)
        x = self.hidden_to_output_layer(x)
        return x

#
# Initialize dataset with tensor instances
#
#dataset = MyDataSet(X, Y)
#-----------------------------------------

#
# Initialize dataset with plain python lists
#
dataset = MyDataSet(x, y)
#-------------------------------------------

dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

print('Showing off DataLoader:')
for x, y in dataloader:
    print(f'DataLoader sample x: {x}, sample y:{y}')

mynet = MyNeuralNet().to(device)

# 
# this is included in parameters()
## print(mynet.input_to_hidden_layer.weight)
#
print('Neural Network Parameters:')
for par in mynet.parameters():
    print(par)

loss_history = []
start_time = time.time()

for _ in range(10000):
    for data in dataloader:
        x, y = data
        mynet.optimizer.zero_grad()
        loss_value = mynet.loss_func(mynet(x), y)
        ## print(f'Loss Value: {loss_value}')
        loss_value.backward()
        mynet.optimizer.step()
        loss_history.append(loss_value.item())

end_time = time.time()
print(f'Time consumed during training: {end_time - start_time}')

plt.plot(loss_history)
plt.title('Loss variation over increasing epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss Value')
plt.show()

# 
# dump parameters after training
#
print('Neural Network Parameters After Training:')
for par in mynet.parameters():
    print(par)

val_x = [[10,11]]
val_x = torch.tensor(val_x).float().to(device)
print(f'checking trained network with {val_x}')
print(f'output for {val_x}: {mynet(val_x)}')


print('okey-dokey, that is all now')

