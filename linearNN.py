import numpy as np

import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
from tqdm import tqdm
from visdom import Visdom


class NNet(nn.Module):
    def __init__(self, p=0.0):
        super().__init__()

        self.lin1 = nn.Linear(28*28, 1000)
        self.lin2 = nn.Linear(1000, 10)

        self.drop_layer = nn.Dropout(p=p)

    def forward(self, x):
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)

        x = self.drop_layer(x)

        return x


BATCH_SIZE = 1000
EPOCHS = 10
LEARNING_RATE = 0.001
DROPOUT = 0.0

# use gpu 0 if cuda is available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

net = NNet(p=DROPOUT)
net.to(device)

#optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE)
optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)

lossfn = nn.CrossEntropyLoss()

T = transforms.Compose([transforms.ToTensor()])

train_data = datasets.MNIST('data', train=True, download=True, transform=T)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False)

test_data = datasets.MNIST('data', train=False, download=True, transform=T)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# values for graphing
train_loss = []
train_acc = []
test_loss = []
test_acc = []

for epoch in range(EPOCHS):
    print(f'epoch: {epoch+1}/{EPOCHS}')

    # training
    net.train()

    # tqdm for progress bar
    for data in tqdm(train_loader, desc='training'):
        # move tensors to gpu if available
        inputs, labels = data[0].to(device), data[1].to(device)

        # change input shape from (28, 28) to (28*28,)
        inputs = inputs.view(-1, 28*28)

        # zero out the gradient
        optimizer.zero_grad()

        outputs = net(inputs)

        # get the predictions for the batch and then check if they're equal to the labels
        prediction = outputs.data.max(1)[1]

        # get value for graphing
        train_acc.append(prediction.eq(labels.data).sum().item() / BATCH_SIZE)

        loss = lossfn(outputs, labels)

        # back propogate
        loss.backward()

        # get value for graphing
        train_loss.append(loss.item())

        # update gradient values
        optimizer.step()

    print(f'\ttrain_loss: {train_loss[-1]:.5f}\n\ttrain_acc: {train_acc[-1]:.4f}')

    # testing
    net.eval()

    # tqdm for progress bar
    for data in tqdm(test_loader, desc='testing'):
        # move tensors to gpu if available
        inputs, labels = data[0].to(device), data[1].to(device)

        # change input shape from (28, 28) to (28*28,)
        inputs = inputs.view(-1, 28*28)

        outputs = net(inputs)

        # get the prediction for the image and then check if it's correct
        prediction = outputs.data.max(1)[1]
        
        # get values for graphing
        test_acc.append(prediction.eq(labels.data).sum().item() / BATCH_SIZE)
        test_loss.append(lossfn(outputs, labels).item())

    print((f'\ttest_loss: {test_loss[-1]:.5f}\n\ttest_acc: {test_acc[-1]:.4f}\n'))

model_name = f'1000hidden__{int(test_acc[-1]*10000)}correct'

# save good models
if test_acc[-1] > .90:
    torch.save(net.state_dict(), f'models/{model_name}.model')

# if visdom server is running, save the matplotlib figure to it
try:
    viz = Visdom()
except:
    viz = None

fig = plt.figure(figsize=(12, 5))

loss_plt = fig.add_subplot(121)
acc_plt = fig.add_subplot(122)

fig.suptitle((f'lr={LEARNING_RATE}, batch_size={BATCH_SIZE}, epochs={EPOCHS}, p={DROPOUT}\n'
              f'final accuracy: {test_acc[-1]:.4f}'))

loss_plt.set_xlabel('epoch')
acc_plt.set_xlabel('epoch')

loss_plt.set_ylabel('loss')
acc_plt.set_ylabel('accuracy')

loss_plt.set_xlim(0, EPOCHS)
acc_plt.set_xlim(0, EPOCHS)

loss_plt.set_ylim(0, max(max(train_loss), max(test_loss)) + .1)
acc_plt.set_ylim(0, 1)

x_train = np.arange(0, EPOCHS, 1 / len(train_loader))
x_test = np.arange(0, EPOCHS, 1 / len(test_loader))

loss_plt.plot(x_train, train_loss, color='b', label='train loss', linewidth=1)
loss_plt.plot(x_test, test_loss, color='g', label='test loss', linewidth=1)

acc_plt.plot(x_train, train_acc, color='b', label='train accuracy', linewidth=1)
acc_plt.plot(x_test, test_acc, color='g', label='test accuracy', linewidth=1)

loss_plt.legend()
acc_plt.legend()

loss_plt.grid()
acc_plt.grid()

if viz:
    viz.matplot(plt)

plt.savefig(f'figs/{model_name}.png')

plt.show()
