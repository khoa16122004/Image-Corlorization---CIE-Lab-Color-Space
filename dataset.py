import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import os
# from config import IMG_SIZE
import torchvision
import numpy as np
from skimage import color

def rgb2lab(img):
    return torch.FloatTensor(np.transpose(color.rgb2lab(np.array(img) / 255.0), (2, 0, 1)))

def simple(img):
    return transforms.ToTensor()(img)


def get_dataset(dataset_name: str):
    if dataset_name == "stl10":
        transform=transforms.Compose([transforms.Lambda(rgb2lab)])
        train_dataset = torchvision.datasets.STL10(root=".Dataset", split='train', 
                                                  download=True, transform=transform)
        test_dataset = torchvision.datasets.STL10(root=".Dataset", split='unlabeled', 
                                                  download=True, transform=transform)

    if dataset_name == 'stl10_simple':
        transform=transforms.Compose([transforms.Lambda(simple)])
        train_dataset = torchvision.datasets.STL10(root=".Dataset", split='train', 
                                                  download=True, transform=transform)
        test_dataset = torchvision.datasets.STL10(root=".Dataset", split='unlabeled', 
                                                  download=True, transform=transform)
    return train_dataset, test_dataset 
    