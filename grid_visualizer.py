import numpy as np
import matplotlib.pyplot as plt
import os, copy
from astropy.io import fits
from astropy.wcs import WCS

# Grid Image Visualizer
class GridVisualizer:
    def __init__(self):
        pass

    def normalize_image(self, image):
        """
        Normalize the image to the range [0, 1].
        """
        image = image.astype(np.float32)
        min_val = np.min(image)
        max_val = np.max(image)
        if max_val > min_val:
            image = (image - min_val) / (max_val - min_val)
        return image
    
    def log_transform(self, image):
        return np.log1p(image)  # log(1 + image) to avoid log(0) issues
    
    def grid_plot(self, image_list, visualize_num, col, 
                  norm=False, log=False, pre_processed=None, super_title=None):        
        space = 0 if visualize_num%col==0 else 1
        plt.figure(figsize=(12, 4*(visualize_num//col+space)))

        if len(image_list) < visualize_num:
            visualize_num = len(image_list)
            
        for i in range(visualize_num):
            # Extract Image Info
            info = os.path.basename(image_list[i]).split('_')
            if pre_processed:
                info = info[-4:-2]
                info[-1] = info[-1]+os.path.splitext(image_list[i])[1]
                
            # Load Image
            if os.path.splitext(image_list[i])[1] == '.fits':
                file = fits.open(image_list[i])
                image = copy.deepcopy(file[0].data)
            elif os.path.splitext(image_list[i])[1] == '.npy':
                image = np.load(image_list[i])

            # Visualize Options
            if norm:
                image = self.normalize_image(image)
            if log:
                image = self.log_transform(image)

            # Add Plot
            plt.subplot(visualize_num//col+space, col, i+1)
            plt.imshow(np.array(image), cmap='gray')
            plt.title(info[-1])
            plt.axis('off')
            if super_title:
                plt.suptitle(super_title+': SSTGC '+info[-2][-6:])

        plt.tight_layout()
        plt.show() 
