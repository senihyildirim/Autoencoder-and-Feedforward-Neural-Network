import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'fashion-mnist', 'utils'))
import mnist_reader

import torch
from torch.utils.data import Dataset

class FashionMNISTFromFile(Dataset):
    def __init__(self, path, kind='train'):
        images, labels = mnist_reader.load_mnist(path, kind)
        self.images = torch.tensor(images / 255.0, dtype=torch.float32).reshape(-1, 1, 28, 28)
        self.labels = torch.tensor(labels).long()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]
