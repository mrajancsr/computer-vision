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

device = "mps"


def generate_noise(size):
    return torch.randn(size, 100).to(device)


@dataclass
class AdvesarialNets:
    generator: nn.Module
    discriminator: nn.Module
    real_samples: DataLoader
    n_epochs: int
    learning_rate: Optional[float] = 0.0002
    beta: Optional[float] = 0.5

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
        self.batch_size = self.real_samples.batch_size
        self.learning_rate = 0.0002  # as specified in paper
        self.beta = 0.5  # specified in the paper
        self.label = torch.full(
            (self.batch_size,), self.real_label, dtype=torch.float, device=device
        )

    def train_discriminator_network(self, real_data, fake_data):
        self.d_optim.zero_grad()
        pred_real = self.discriminator(real_data)
        loss_real = self.loss_fn(pred_real.squeeze(), self.label)
        loss_real.backward()

        # compute loss on making prediction on fake data
        pred_fake = self.discriminator(fake_data)
        self.label.fill_(self.fake_label)
        loss_fake = self.loss_fn(pred_fake.squeeze(), self.label)
        loss_fake.backward()

        self.d_optim.step()
        # return back label to good labels
        self.label.fill_(self.real_label)
        return loss_real.item() + loss_fake.item()

    def train_generator_network(self, fake_data):
        self.g_optim.zero_grad()
        pred_fake = self.discriminator(fake_data)
        loss = self.loss_fn(pred_fake.squeeze(), self.label)
        loss.backward()
        self.g_optim.step()
        return loss.item()

    def train(self):
        log = ts.Report(self.n_epochs)
        n = len(self.real_samples)
        for epoch in range(self.n_epochs):
            for idx, (xbatch, ybatch) in enumerate(self.real_samples):
                # better to do this step in a dataset preprocessing
                real_data = xbatch.view(len(xbatch), -1).to(device)
                noise = generate_noise(len(real_data)).to(device)
                fake_data = self.generator(noise).to(device)
                # create fresh tensor so that when backward() is called in discriminator function, it doesn't affect
                # tensors in the generator which creates fakes data
                # and finally train discriminator
                dloss = self.train_discriminator_network(real_data, fake_data.detach())

                # train generator
                # generate new set of fake images from noisy data and train generator
                noise = generate_noise(len(real_data))
                fake_data = self.generator(noise).to(device)
                gloss = self.train_generator_network(fake_data)
                log.record(epoch + (1 + idx) / n, d_loss=dloss, g_loss=gloss, end="\r")
            log.report_avgs(epoch + 1)

        log.plot_epochs(["d_loss", "g_loss"])
