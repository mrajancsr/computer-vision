# The purpose of this file is to fit a Generative Advesarial Network to MNIST dataset
import glob
import os
import random

import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary
from torchvision import transforms
from torchvision.datasets import MNIST

from neptunevision.generative_advesarial_network.gann import AdvesarialNets

DATASET_PATH = os.path.join(os.getcwd(), "datasets")


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Faces(Dataset):
    def __init__(self, folder, transform=None):
        self.file_list = glob.glob(folder + "/cropped_faces/*.jpg")

        random.shuffle(self.file_list)
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img = Image.open(self.file_list[idx])
        if self.transform is not None:
            img = self.transform(img)
        return img


# Classifier for DCGAN
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64 * 2, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64 * 4, 64 * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64 * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )
        self.apply(weights_init)


# Generates an image out of random noise of size (1,100)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(100, 64 * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(64 * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * 8, 64 * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * 4, 64 * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64 * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(64 * 2, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )
        self.apply(weights_init)

    def forward(self, x):
        return self.model(x)


def main():
    device = "mps"

    # Normalize and convert data to tensors
    transform = transforms.Compose(
        [
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    # Download the faces dataset
    folder = os.path.join(DATASET_PATH, 'male_female_faces')
    mnist = MNIST(DATASET_PATH, download=True, train=True, transform=transform)
    mnist_dl = DataLoader(mnist, batch_size=128, shuffle=True, drop_last=True)

    discriminator = Discriminator()
    generator = Generator()
    print("\n")
    print("Summary of Discriminator")
    summary(discriminator, torch.zeros(1, 784))

    print("\n")
    print("Summary of Generator")
    summary(generator, torch.zeros(1, 100))
    print("\n")

    # Number of training epochs
    num_epochs = 200

    gan = AdvesarialNets(
        generator.to(device), discriminator.to(device), mnist_dl, num_epochs
    )
    gan.train()


if __name__ == "__main__":
    main()
