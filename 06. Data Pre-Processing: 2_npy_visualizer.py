import numpy as np
import matplotlib.pyplot as plt
import os, time, glob, copy

# ------- Edit it to recieve variables from the config.py ------- #

root_dir = os.getcwd() # /home/jiwon/UKIDSSxIRAC

survey1 = 'ukidss'
filters1 = ['J', 'H', 'K']

survey2 = 'spitzer'
filters2 = ['ch1', 'ch2', 'ch3', 'ch4']

train_ratio = 0.6
val_ratio = 0.2
test_ratio = 0.2

augmentation = ['VF', 'HF', 'Rot90', 'Rot180', 'Rot270']

# --------------------------------------------------------------- #

targetlist = np.loadtxt(os.path.join(root_dir, 'targetlist.txt'), dtype='str')

class Visualizer:
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

    def visualize_npy_files(self, filename):
        image = np.load(filename)
        image = self.normalize_image(image)
        image = self.log_transform(image)  

        plt.figure()        

        if image.ndim == 3:  # Assuming the shape is (channels, height, width)
            channels, height, width = image.shape
            fig, axes = plt.subplots(1, channels, figsize=(5*channels, 5))

            for i in range(channels):
                axes[i].imshow(image[i], origin='lower', cmap='gray')
                axes[i].axis('off')
                if len(channel_name) == channels:
                    axes[i].set_title(f'{channel_name[i]}')

        elif image.ndim == 2:  # For grayscale images
            plt.imshow(image, origin='lower', cmap='gray')
            plt.axis('off')
            plt.title(f'{channel_name[0]}')
        
        plt.show()


data_dir = root_dir+'/data/merged'
channel_name_X = ['J', 'H', 'K', 'ch2', 'ch3', 'ch4']
channel_name_y = ['ch1']
visualizer = Visualizer()

for target in targetlist[0:1]:
    print(target)
    visualizer.visualize_npy_files(os.path.join(data_dir,target+'_X.npy'), channel_name_X)
    visualizer.visualize_npy_files(os.path.join(data_dir,target+'_y.npy'), channel_name_y)
    visualizer.visualize_npy_files(os.path.join(root_dir,'data','ukidss','rectified_'+target+'_K.npy'), ['K'])

plt.show()
