# The purpose of this file is to fit a Generative Advesarial Network to MNIST dataset
from typing import Optional
from torchvision.datasets import MNIST
from torchvision import transforms
import pandas as pd
import torch
from torchsummary import summary

import numpy as np
from dataclasses import dataclass
import torch_snippets as ts
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn


import os

device = "mps"

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))],
)
path = os.path.join(os.getcwd(), "datasets")
# mnist = MNIST(path, download=True, train=True, transform=transform)


def make_layer(in_channel: int, out_channel):
    """Makes a layer in the Discriminator network

    Parameters
    ----------
    in_channel : int
        _description_
    out_channel : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    return nn.Sequential(nn.Linear(in_channel, out_channel), nn.LeakyReLU(0.2))


# Classifier for GAN
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            make_layer(784, 1024),
            make_layer(1024, 512),
            make_layer(512, 256),
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
            nn.Linear(1024, 784),
            nn.Tanh(),
        )

    def forward(self, x):
        return self.model(x)


@dataclass
class AdvesarialNets:
    generator: Generator
    discriminator: Discriminator
    real_samples: DataLoader
    n_epochs: int
    learning_rate: Optional[float] = 1e-3

    def __post_init__(self):
        self.g_optim: torch.optim = torch.optim.Adam(
            self.generator.parameters(), self.learning_rate
        )
        self.d_optim: torch.optim = torch.optim.Adam(
            self.discriminator.parameters(), self.learning_rate
        )
        self.loss_fn = nn.BCELoss()
        self.real_label = 1
        self.fake_label = 0

    def train_discriminator_network(self, real_data, fake_data):
        self.d_optim.zero_grad()
        pred_real = self.discriminator(real_data)
        loss_real = self.loss_fn(pred_real.squeeze(), self.real_label)
        loss_real.backward()

        # compute loss on making prediction on fake data
        pred_fake = self.discriminator(fake_data)
        loss_fake = self.loss_fn(pred_fake, self.fake_label)
        loss_fake.backward()

        self.d_optim.step()
        return loss_real.item() + loss_fake.item()

    def train_generator_network(self, fake_data):
        self.g_optim.zero_grad()
        pred_fake = self.discriminator(fake_data)
        loss = self.loss_fn(pred_fake.squeeze(), self.real_label)
        loss.backward()
        self.g_optim.step()
        return loss.item()

    def train(self):
        for epoch in range(self.n_epochs):
            for idx, (xbatch, ybatch) in enumerate(self.real_samples):
                # better to do this step in a dataset preprocessing
                real_data = xbatch.view(len(xbatch), -1).to(device)
                noise = generate_noise(len(real_data))
                fake_data = generator(noise).to(device)
                # create fresh tensor so that when backward() is called in discriminator function, it doesn't affect
                # tensors in the generator which creates fakes data
                fake_data = fake_data.detach()
                # train discriminator
                dloss = self.train_discriminator_network(real_data, fake_data)
                # train generator
                # generate new set of fake images from noisy data and train generator
                noise = generate_noise(len(real_data))
                fake_data = generator(noise).to(device)
                gloss = self.train_generator_network(fake_data)
                log.record(epoch + (1 + idx) / n, d_loss=dloss, g_loss=gloss, end="\r")
            log.report_avgs(epoch + 1)

        log.plot_epochs(["d_loss", "g_loss"])


def main():
    # mnist = MNIST(path, download=True, train=True, transform=transform)
    # mnist_dl = DataLoader(mnist, batch_size=128, shuffle=True, drop_last=True)

    discriminator = Discriminator().to(device)
    generator = Generator().to(device)
    print("\n")
    print("Summary of Discriminator")
    summary(discriminator, torch.zeros(1, 784))

    print("\n")
    print("Summary of Generator")
    summary(generator, torch.zeros(1, 100))
    print("\n")


if __name__ == "__main__":
    main()
