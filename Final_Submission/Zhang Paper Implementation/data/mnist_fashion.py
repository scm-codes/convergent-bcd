import numpy as np
from torchvision.datasets import FashionMNIST


def mnist_fashion():
    ds = FashionMNIST("\\fashion-mnsit\\data", download=True)
    x = np.reshape(ds.data.numpy(), (len(ds.data), -1, 1))
    avg = np.mean(x)
    std = np.std(x)
    x = (x - avg) / std
    y = ds.targets.numpy()
    num_labels = len(np.unique(y))
    labels = np.arange(num_labels)
    y = labels[y]
    onehot_encode = np.eye(num_labels)
    y = onehot_encode[y][..., np.newaxis]
    return x, y
