import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

batch_size = 4
import os

path = os.path.join(os.getcwd(), "cv_with_pytorch", "datasets", "CIFAR")

trainset = torchvision.datasets.CIFAR10(
    root=path, train=True, download=True, transform=transform
)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.CIFAR10(
    root=path, train=False, download=True, transform=transform
)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

classes = (
    "plane",
    "car",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 128, kernel_size=3)
        self.max_pool = nn.MaxPool2d(2, 2)
        self.activation = nn.ReLU()
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3)
        self.conv3 = nn.Conv2d(256, 512, kernel_size=3)
        self.conv4 = nn.Conv2d(512, 256, kernel_size=3)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(1024, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 10)

    def forward(self, x):
        # Implement the forward pass with using the layers defined above
        # and the proper activation functions
        x = self.conv1(x)
        x = self.max_pool(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.max_pool(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        return x


net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(), lr=0.0001)

device = "mps"


def train_network(net, n_epochs=2):
    net.to(device)

    for epoch in range(n_epochs):  # loop over the dataset multiple times
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            if i % 1000 == 0:
                print(f"Epoch={epoch + 1} Iter={i + 1:5d} Loss={loss.item():.3f}")
                running_loss = 0.0
    print("Finished Training")
    return net


for x_batch, y_batch in trainloader:
    break

net.forward(x_batch)
