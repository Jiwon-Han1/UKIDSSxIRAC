### Saved as ~/UKIDSSxIRAC/1_prep_stacking_and_augmentation.py

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

targetlist = np.loadtxt(os.path.join(root_dir, 'targetlist.txt'), dtype='str')

train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2

augmentation = ['VF', 'HF', 'Rot90', 'Rot180', 'Rot270', 'VF_Rot90', 'HF_Rot90']

verbose_fits_to_npy = False
verbose_image_stacking = True
verbose_augmentation = False

# --------------------------------------------------------------- #
if not os.path.exists(root_dir+'/data/merged'):
    print(f'\n=== Making "merged" Directory ===\n')
    os.makedirs(root_dir+'/data/merged')

save_dir = root_dir+'/data/merged'

class ImageStacker:
    def __init__(self, save_dir, channels_order=None):
        self.save_dir = save_dir
        self.channels_order = channels_order 

    def load_and_stack(self, target_datalist):
        channel_data = {}
        for target_data in target_datalist:
            band = os.path.splitext(target_data)[0].split('_')[-1]
            channel_data[band] = np.load(target_data)

        if self.channels_order is None:
            self.channels_order = ['J', 'H', 'K', 'ch2', 'ch3', 'ch4']

        y_data = channel_data.pop('ch1')
        X_data = np.stack([channel_data[ch] for ch in self.channels_order], axis=0)
        return X_data, y_data
    
    def save_as_Xy(self, target, target_datalist, verbose=False):
        X_data, y_data = self.load_and_stack(target_datalist)
        if verbose:
            print('\n'+'[[ # '+target+' ]]')
            print(np.shape(X_data), os.path.join(self.save_dir,f'{target}_X.npy'))
            print(np.shape(y_data), os.path.join(self.save_dir,f'{target}_y.npy'))
        np.save(os.path.join(self.save_dir,f'{target}_X.npy'), X_data)
        np.save(os.path.join(self.save_dir,f'{target}_y.npy'), y_data)

class ImageAugmentation: # To call getattr
    def __init__(self):
        pass

    def VF(self, image_array): # vertical_flip
        return np.flipud(image_array)

    def HF(self, image_array): # horizontal_flip
        return np.fliplr(image_array)

    def Rot90(self, image_array): # rotate_90
        # return np.rot90(image_array, k=1, axes=(image_array.ndim-2, image_array.ndim-1))
        return np.rot90(image_array, k=1, axes=(0, 1))

    def Rot180(self, image_array): # rotate_180
        # return np.rot90(image_array, k=2, axes=(image_array.ndim-2, image_array.ndim-1))
        return np.rot90(image_array, k=2, axes=(0, 1))
    
    def Rot270(self, image_array): # rotate_270
        # return np.rot90(image_array, k=3, axes=(image_array.ndim-2, image_array.ndim-1))
        return np.rot90(image_array, k=3, axes=(0, 1))
    
    def VF_Rot90(self, image_array): # rotate_90_flipped_VF
        return np.rot90(np.flipud(image_array), k=1, axes=(0, 1)) 

    def HF_Rot90(self, image_array): # rotate_90_flipped_HF
        return np.rot90(np.fliplr(image_array), k=1, axes=(0, 1)) 
    
    def augment_and_visualize(self, image_array):
        # Perform augmentations
        augments = {
            'Original': image_array,
            'Vertical Flip': self.VF(image_array),
            'Horizontal Flip': self.HF(image_array),
            'Rotate 90': self.Rot90(image_array),
            'Rotate 180': self.Rot180(image_array),
            'Rotate 270': self.Rot270(image_array),
            'Vertical Flip + Rotate 90': self.VF_Rot90(image_array),
            'Horizontal Flip + Rotate 90': self.HF_Rot90(image_array)
        }

        # Visualize the results
        plt.figure(figsize=(12, 3*(len(augments)//3+1)))
        for i, (title, aug_image) in enumerate(augments.items()):
            plt.subplot(len(augments)//3+1, 3, i + 1)
            plt.imshow(aug_image)
            plt.title(title)
            plt.axis('off')
        plt.tight_layout()
        plt.show()

img_stacker = ImageStacker(save_dir)
augmentor = ImageAugmentation()

# if not os.path.exists(root_dir+'/data_3_channels'):
#     print(f'\n=== Making "3-channel merged" Directory ===\n')
#     os.makedirs(root_dir+'/data_3_channels')
# img_stacker = ImageStacker(root_dir+'/data_3_channels', channels_order=['H','K','ch2'])

img_stacker = ImageStacker(save_dir)
augmentor = ImageAugmentation()

for target in targetlist:
    target_datalist = [os.path.abspath(x) for x in glob.glob(os.path.join(root_dir,'data','**', f'*{target}*'), recursive=True)]
    img_stacker.save_as_Xy(target, target_datalist, verbose=verbose_image_stacking)

    if len(augmentation) > 0:
        X_data = np.load(os.path.join(save_dir,f'{target}_X.npy'))
        y_data = np.load(os.path.join(save_dir,f'{target}_y.npy'))

        for aug in augmentation:
            aug_data_X = getattr(augmentor, aug)(X_data)
            aug_data_y = getattr(augmentor, aug)(y_data)
            
            np.save(os.path.join(save_dir, f"{target}_{aug}_X.npy"), aug_data_X)
            np.save(os.path.join(save_dir, f"{target}_{aug}_y.npy"), aug_data_y)
            
            if verbose_augmentation:
                print(os.path.join(save_dir,f"{target}_{aug}_X.npy"))
                print(os.path.join(save_dir,f"{target}_{aug}_y.npy"))

### Example (Dog)
image_path = os.path.join(root_dir,'dog.png')

image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

augmentor = ImageAugmentation()
augmentor.augment_and_visualize(image)


            





    

###### Do Merging the Augmentation to the ImageStacker Class!
