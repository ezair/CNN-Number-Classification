"""Program That builds a convolutional NN for classifying a number (0 - 10).
The model is the outputted to a binary pickle file so that it can be used again
in the future.
"""

import os

import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim
import torch.nn as nn

from classify.model import ConvNetwork
from classify.train_test import train, test


def main():
    batch_size = 64

    transform_normalizer = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))]
    )

    # Get training and testing set (built into torch already).
    training_set = datasets.MNIST(
        root="dataset/", train=True, download=True, transform=transform_normalizer
    )
    testing_set = datasets.MNIST(
        root="dataset/", train=False, download=True, transform=transform_normalizer
    )
    train_loader = torch.utils.data.DataLoader(
        training_set, shuffle=True, batch_size=batch_size
    )
    test_loader = torch.utils.data.DataLoader(
        testing_set, shuffle=True, batch_size=batch_size
    )

    model = ConvNetwork()

    train(
        model=model,
        training_set=train_loader,
        optimizer=optim.Adam(model.parameters(), lr=0.001),
        loss_function=nn.CrossEntropyLoss(),
        epochs=5,
    )

    test(testing_set=test_loader, model=model)

    # Save model to file encoded file.
    torch.save(
        model,
        os.cwd() + "models/ConvNetwork.pt",
    )


if __name__ == "__main__":
    main()
