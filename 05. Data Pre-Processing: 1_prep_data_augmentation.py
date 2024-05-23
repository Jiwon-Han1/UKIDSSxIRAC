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

targetlist = np.loadtxt(os.path.join(root_dir, 'targetlist.txt'), dtype='str')

if not os.path.exists(root_dir+'/data/merged'):
    print(f'\n=== Making "merged" Directory ===\n')
    os.makedirs(root_dir+'/data/merged')

class ImageStacker:
    def __init__(self, save_dir, channels_order=None):
        self.save_dir = save_dir
        self.channels_order = channels_order 

    def load_and_stack(self, target_datalist):
        channel_data = {}
        for target_data in target_datalist:
            filter = os.path.splitext(target_data)[0].split('_')[-1]
            channel_data[filter] = np.load(target_data)

        if self.channels_order is None:
            self.channels_order = ['J', 'H', 'K', 'ch2', 'ch3', 'ch4']

        y_data = channel_data.pop('ch1')
        X_data = np.stack([channel_data[ch] for ch in self.channels_order], axis=0)
        return X_data, y_data
    
    def save_as_Xy(self, target, target_datalist, verbose=False):
        X_data, y_data = self.load_and_stack(target_datalist)######
        if verbose:
            print('\n'+'[[ # '+target+' ]]')
            print(np.shape(X_data), os.path.join(self.save_dir,f'{target}_X.npy'))
            print(np.shape(y_data), os.path.join(self.save_dir,f'{target}_y.npy'))
        np.save(os.path.join(self.save_dir,f'{target}_X.npy'), X_data)
        np.save(os.path.join(self.save_dir,f'{target}_y.npy'), y_data)

save_dir = root_dir+'/data/merged'
img_stacker = ImageStacker(save_dir)#, channels_order=['H','K','ch2'])

for target in targetlist:
    target_datalist = [os.path.abspath(x) for x in glob.glob(os.path.join(root_dir,'data','**', f'*{target}*'), recursive=True)]
    img_stacker.save_as_Xy(target, target_datalist)#, verbose=True)
    

###### Do Merging the Augmentation to the ImageStacker Class!

class ImageAugmentation:
    def __init__(self):
        pass

    def vertical_flip(self, image_array):
        return np.flipud(image_array)

    def horizontal_flip(self, image_array):
        return np.fliplr(image_array)

    def rotate_90(self, image_array):
        return np.rot90(image_array, k=1, axes=(0, 1))

    def rotate_180(self, image_array):
        return np.rot90(image_array, k=2, axes=(0, 1))
    
    def rotate_270(self, image_array):
        return np.rot90(image_array, k=3, axes=(0, 1))

a = np.arange(4).reshape((2,2))

augmentor = ImageAugmentation()
# print(a)
# print(augmentor.horizontal_flip(a))
# print(augmentor.vertical_flip(a))
# print(augmentor.rotate_90(a))
# print(augmentor.rotate_180(a))
# print(augmentor.rotate_270(a))
