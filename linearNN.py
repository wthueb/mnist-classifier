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
        x = F.relu(self.lin1(x))
        x = self.lin2(x)
        x = self.drop_layer(x)

        return x


BATCH_SIZE = 100
EPOCHS = 100
LEARNING_RATE = 0.001
DROPOUT = 0.0

T = transforms.Compose([transforms.ToTensor()])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

net = NNet(p=DROPOUT)
net.to(device)

optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE)

lossfn = nn.CrossEntropyLoss()

train_data = datasets.MNIST('data', train=True, download=True, transform=T)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=False)

test_data = datasets.MNIST('data', train=False, download=True, transform=T)
test_loader = torch.utils.data.DataLoader(test_data, shuffle=False)

# values for graphing
train_loss_vals = []
train_acc_vals = []
test_loss_vals = []
test_acc_vals = []

for epoch in range(EPOCHS):
    print(f'epoch: {epoch+1}/{EPOCHS}')

    # training
    net.train()

    train_loss = 0
    train_acc = 0

    # tqdm for progress bar
    for data in tqdm(train_loader, desc='training'):
        # move tensors to gpu if available
        inputs, labels = data[0].to(device), data[1].to(device)

        # change input from (28, 28) to (28,)
        inputs = inputs.view(-1, 28*28)

        optimizer.zero_grad()

        outputs = net(inputs)

        prediction = outputs.data.max(1)[1]
        train_acc += prediction.eq(labels.data).sum().item()

        loss = lossfn(outputs, labels)

        loss.backward()

        train_loss += loss.item()

        optimizer.step()

    train_loss /= len(train_loader)
    train_acc /= len(train_loader.dataset)

    train_loss_vals.append(train_loss)
    train_acc_vals.append(train_acc)

    print(f'\ttrain_loss: {train_loss:.5f}\n\ttrain_acc: {train_acc:.4f}')

    # testing
    net.eval()

    test_loss = 0
    test_acc = 0

    # tqdm for progress bar
    for data in tqdm(test_loader, desc='testing'):
        # move tensors to gpu if available
        inputs, labels = data[0].to(device), data[1].to(device)

        # change input from (28, 28) to (28,)
        inputs = inputs.view(-1, 28*28)

        outputs = net(inputs)

        prediction = outputs.data.max(1)[1]
        test_acc += prediction.eq(labels.data).sum().item()

        test_loss += lossfn(outputs, labels).item()

    test_loss /= len(test_loader)
    test_acc /= len(test_loader)

    test_loss_vals.append(test_loss)
    test_acc_vals.append(test_acc)

    print((f'\ttest_loss: {test_loss:.5f}\n\ttest_acc: {test_acc:.4f}\n'))

if test_acc_vals[-1] > .90:
    model_name = f'1000hidden__{int(test_acc_vals[-1]*10000)}correct.model'
    torch.save(net.state_dict(), f'models/{model_name}')

try:
    viz = Visdom()
except:
    viz = None

fig = plt.figure(figsize=(12, 5))

loss_plt = fig.add_subplot(121)
acc_plt = fig.add_subplot(122)

fig.suptitle((f'lr={LEARNING_RATE}, batch_size={BATCH_SIZE}, epochs={EPOCHS}, p={DROPOUT}\n'
              f'final accuracy: {test_acc_vals[-1]:.4f}'))

loss_plt.set_xlabel('epoch')
acc_plt.set_xlabel('epoch')

loss_plt.set_ylabel('loss')
acc_plt.set_ylabel('accuracy')

loss_plt.set_xlim(0, EPOCHS + 1)
acc_plt.set_xlim(0, EPOCHS + 1)

loss_plt.set_ylim(0, max(train_loss_vals + test_loss_vals) + .1)
acc_plt.set_ylim(0, 1)

loss_plt.plot(range(1, EPOCHS + 1), train_loss_vals, color='b', label='train loss', linewidth=1)
loss_plt.plot(range(1, EPOCHS + 1), test_loss_vals, color='g', label='test loss', linewidth=1)

acc_plt.plot(range(1, EPOCHS + 1), train_acc_vals, color='b', label='train accuracy', linewidth=1)
acc_plt.plot(range(1, EPOCHS + 1), test_acc_vals, color='g', label='test accuracy', linewidth=1)

loss_plt.legend()
acc_plt.legend()

loss_plt.grid()
acc_plt.grid()

if viz:
    viz.matplot(plt)

plt.show()
