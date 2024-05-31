import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import os, time, glob, copy

from astropy.io import fits
import cv2

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms

# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import torchvision
# from torchvision import datasets, transforms, utils

# ------- Edit it to recieve variables from the config.py ------- #

root_dir = os.getcwd() # /home/jiwon/UKIDSSxIRAC

survey1 = 'ukidss'
filters1 = ['J', 'H', 'K']

survey2 = 'spitzer'
filters2 = ['ch1', 'ch2', 'ch3', 'ch4']

train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2

# --------------------------------------------------------------- #

print(os.getcwd())
targetlist = np.loadtxt(os.path.join(root_dir, 'targetlist.txt'), dtype='str')
print(targetlist)

class MyDataSet(Dataset):
    def __init__(self, datalist, transform=None):
        self.datalist = datalist ### edit
        self.transform = transform
        pass

    def __len__(self):  # Return the Dataset Size
        return(len(self.datalist)) ### edit

    def __getitem__(self):  # Load Images and Apply the Transform

        pass

transform = transforms
