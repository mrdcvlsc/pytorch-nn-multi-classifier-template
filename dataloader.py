import torch

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt

LABEL_MAP = {
    0: "Zero",
    1: "One",
    2: "Two",
    3: "Three",
    4: "Four",
    5: "Five",
    6: "Six",
    7: "Seven",
    8: "Eight",
    9: "Nine",
}
"""
MNIST DATABASE DIGIT LABEL MAP
"""

if __name__ == "__main__":

    # If you are going to address your own problem, you would first need to create your own 'Dataset' class.
    # However, for the sake of simplicity, we will use the pre-existing 'Dataset' class from the MNIST database.
    # MNIST is a dataset consisting of grayscale images of handwritten digits.

    training_data = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=ToTensor()
    )

    test_data = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=ToTensor()
    )

    # we load our "Dataset" using the 'DataLoader' class of pytorch
    train_dataloader = DataLoader(training_data, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=64, shuffle=True)

    # Display image and label.
    train_inputs, train_labels = next(iter(train_dataloader))

    train_inputs: torch.Tensor = train_inputs
    train_labels: torch.Tensor = train_labels

    img = train_inputs[0].squeeze()
    label = train_labels[0]

    print(f"Input Features batch shape : {train_inputs.size()}")
    print(f"Labels batch shape         : {train_labels.size()}\n")

    print(f"Input Features batch dtype : {train_inputs.dtype}")
    print(f"Labels batch dtype         : {train_labels.dtype}\n")

    print(f"Label: {LABEL_MAP[int(label.item())]}")

    plt.imshow(img, cmap="gray")
    plt.show()

    # Input Features batch shape : torch.Size([64, 1, 28, 28])
    # Labels batch shape         : torch.Size([64])

    # Input Features batch dtype : torch.float32
    # Labels batch dtype         : torch.int64
