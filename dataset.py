# dataset/dataset.py
import sys
import numpy as np
import os
from torch.utils.data import Dataset

# Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, image_dir, transform=None, inference=False):
        self.image_dir = image_dir
        self.transform = transform
        self.inference = inference
        
    def __len__(self):
        return len(self.image_dir)

    def __getitem__(self, idx):
        data = self.image_dir[idx]
        X_image = np.load(data['X'])
        y_image = np.load(data['y'])  
        # print('Numpy Array Shape: ', X_image.shape)  # (C, H, W): (6, 300, 300)

        if self.transform:
            X_image = X_image.transpose(1, 2, 0)  # (C, H, W) --> (H, W, C)
            if y_image.ndim == 3:  # (C, H, W)
                y_image = y_image.transpose(1, 2, 0)  # (C, H, W) --> (H, W, C)
            elif y_image.ndim == 2:  # (H, W)
                y_image = y_image[:, :, np.newaxis]  # (H, W) --> (H, W, 1)

            X_image = self.transform(X_image)
            y_image = self.transform(y_image)
            # print('Tensor Shape: ', X_image.shape)  # (C, H, W): (6, 300, 300)

        if self.inference:
            target_name = os.path.splitext(os.path.basename(data['X']))[0].split('_')[0]
            return X_image, y_image, target_name
        else:
            return X_image, y_image ###
