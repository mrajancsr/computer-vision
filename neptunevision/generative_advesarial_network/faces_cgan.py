"""The purpose of this module is to train the computer to generate fake faces of people
   by giving age and gender as inputs.  The program uses Conditional Generative Advesarial Networks
   to generate an image given the inputs
"""

import glob
import os

import cv2
import torch
import torch.nn as nn
import torch_snippets as ts
from gan import ConditionalAdvesarialNets
from torch.utils.data import DataLoader, Dataset
from torchsummary import summary
from torchvision import transforms
from torchvision.utils import make_grid

DATASET_PATH = os.path.join(os.getcwd(), "datasets")


class Faces(Dataset):
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )

    def __init__(self, root_folder, transforms=None):
        self.folder = os.path.join(DATASET_PATH, root_folder)
        self.crop_images("males")
        self.crop_images("females")

    def crop_images(self, folder_name: str, destination_folder: str = "cropped_faces"):
        """Crops images from folder_name located in faces folder
        and stores results given by destination_folder

        Parameters
        ----------
        folder_name : str
            folder name to crop faces from, one of males/females
        destination_folder : str, optional, default='cropped_faces'
            writes cropped faces into this folder
        """
        file_list = glob.glob(self.folder + "/" + folder_name + "*.jpg")
        destination_folder = os.path.join(self.folder, destination_folder, folder_name)
        for idx, file in enumerate(file_list):
            img = ts.read(file, 1)
            grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = Faces.face_cascade.detectMultiScale(grey, 1.3, 5)
            for x, y, w, h in faces:
                img2 = img[y : (y + h), x : (x + w), :]
            write_to_path = os.path.join(destination_folder, f"{idx}.jpg")
            cv2.imwrite(write_to_path, cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))

    def __getitem__(self, ix):
        pass

    def __len__(self):
        pass


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
    num_epochs = 10

    cgan = ConditionalAdvesarialNets(
        generator.to(device), discriminator.to(device), mnist_dl, num_epochs
    )
    cgan.train()

    # generate 64 batches of random samples to generate 64 images
    z = torch.randn(64, 100).to(device)
    # each of the 64 images are 1x28x28 (grey scale)
    sample_images = generator(z).data.cpu().view(64, 1, 28, 28)
    grid = make_grid(sample_images, nrow=8, normalize=True)
    ts.show(grid.cpu().detach().permute(1, 2, 0), sz=5)


if __name__ == "__main__":
    main()
