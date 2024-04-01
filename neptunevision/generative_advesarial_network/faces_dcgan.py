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

from neptunevision.generative_advesarial_network.gann import AdvesarialNets

DATASET_PATH = os.path.join(os.getcwd(), "datasets")


def make_convolutional_layer(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    padding: int | str = 0,
    bias: bool = False,
) -> nn.Sequential:
    """Makes a Layer composed of 2D Convolution, BatchNorm and LeakyReLU

    Parameters
    ----------
    in_features : int
        the number of features in sample
    out_features : _type_
        the number of output units in sample

    Returns
    -------
    nn.Sequential
        Sequential Layer composed of Linear and a LeakyReLU for activation
    """
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(0.2, inplace=True),
    )


def make_convtranspose2d_layer(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int,
    padding: int | str = 0,
    bias: bool = False,
) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size, stride, padding, bias=bias
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(True),
    )


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
            make_convolutional_layer(64, 64 * 2, 4, 2, 1, bias=False),
            make_convolutional_layer(64 * 2, 64 * 4, 4, 2, 1, bias=False),
            make_convolutional_layer(64 * 4, 64 * 8, 4, 2, 1, bias=False),
            nn.Conv2d(64 * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )
        self.apply(weights_init)


# Generates an image out of random noise of size (1,100)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            make_convtranspose2d_layer(100, 64 * 8, 4, 1, 0, bias=False),
            make_convtranspose2d_layer(64 * 8, 64 * 4, 4, 2, 1, bias=False),
            make_convtranspose2d_layer(64 * 4, 64 * 2, 4, 2, 1, bias=False),
            make_convtranspose2d_layer(64 * 2, 64, 4, 2, 1, bias=False),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh(),
        )
        self.apply(weights_init)

    def forward(self, x):
        return self.model(x)


def main():
    device = "mps"

    BATCH_SIZE = 64
    NUM_WORKERS = 8
    # Resize image to 3x64x64 and normalize
    transform = transforms.Compose(
        [
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )
    # Download the faces dataset
    folder = os.path.join(DATASET_PATH, "male_female_faces")
    faces_ds = Faces(folder=folder, transform=transform)
    faces_dl = DataLoader(
        faces_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS
    )

    discriminator = Discriminator()
    print("\n")
    print("Summary of Discriminator")
    summary(discriminator, torch.zeros(1, 3, 64, 64))

    print("\n")
    generator = Generator()
    print("Summary of Generator")
    summary(generator, torch.zeros(1, 100, 1, 1))
    print("\n")

    # Number of training epochs
    num_epochs = 20

    gan = AdvesarialNets(
        generator.to(device), discriminator.to(device), faces_dl, num_epochs
    )
    gan.train()


if __name__ == "__main__":
    main()
