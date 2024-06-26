# The purpose of this file is to fit a Generative Advesarial Network to MNIST dataset
import os

import torch
import torch.nn as nn
import torch_snippets as ts
from torch.utils.data import DataLoader
from torchsummary import summary
from torchvision import transforms
from torchvision.datasets import MNIST
from torchvision.utils import make_grid

from neptunevision.generative_advesarial_network.gann import AdvesarialNets

DATASET_PATH = os.path.join(os.getcwd(), "datasets")


def make_layer(in_features: int, out_features) -> nn.Sequential:
    """Makes a Sequential Layer with a Leaky ReLU for activation

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
    return nn.Sequential(nn.Linear(in_features, out_features), nn.LeakyReLU(0.2))


# Classifier for GAN
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            make_layer(784, 1024),
            nn.Dropout(0.3),
            make_layer(1024, 512),
            nn.Dropout(0.3),
            make_layer(512, 256),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, input_image):
        return self.model(input_image)


# Generates an image out of random noise of size (1,100)
class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            make_layer(100, 256),
            make_layer(256, 512),
            make_layer(512, 1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 784),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.model(x)


def main():
    device = "mps"

    # Normalize and convert data to tensors
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))]
    )
    # Download the mnist dataset and create dataloader
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

    # generate 64 batches of random samples to generate 64 images
    z = torch.randn(64, 100).to(device)
    # each of the 64 images are 1x28x28 (grey scale)
    sample_images = generator(z).data.cpu().view(64, 1, 28, 28)
    grid = make_grid(sample_images, nrow=8, normalize=True)
    ts.show(grid.cpu().detach().permute(1, 2, 0), sz=5)


if __name__ == "__main__":
    main()
