import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
import os
# from config import IMG_SIZE
import torchvision
import numpy as np
from skimage import color

def import_image(img):
    return torch.FloatTensor(np.transpose(color.rgb2lab(np.array(img) / 255.0), (2, 0, 1)))

def get_dataset(dataset_name: str, transform=transforms.Compose([transforms.Lambda(import_image)])):
    if dataset_name == "stl10":
        train_dataset = torchvision.datasets.STL10(root=".Dataset", split='train', 
                                                  download=True, transform=transform)
        test_dataset = torchvision.datasets.STL10(root=".Dataset", split='unlabeled', 
                                                  download=True, transform=transform)

    return train_dataset, test_dataset 
    