import numpy as np
import matplotlib.pyplot as plt

class ImageAugmentation: # To call getattr
    def __init__(self):
        pass

    def VF(self, image_array): # vertical_flip
        if image_array.ndim == 2:  # Single 2D image
            flipped = np.flipud(image_array)
        else:
            flipped =  np.array([np.flipud(channel) for channel in image_array])
        return flipped

    def HF(self, image_array): # horizontal_flip
        if image_array.ndim == 2:  # Single 2D image
            return np.fliplr(image_array)
        else:
            return np.array([np.fliplr(channel) for channel in image_array])
        
    def Rot90(self, image_array): # rotate_90
        if image_array.ndim == 2:  # Single 2D image
            return np.rot90(image_array, k=1)
        else:  # Multi-channel image
            return np.rot90(image_array, k=1, axes=(1, 2))

    def Rot180(self, image_array): # rotate_180
        if image_array.ndim == 2:  # Single 2D image
            return np.rot90(image_array, k=2)
        else:  # Multi-channel image
            return np.rot90(image_array, k=2, axes=(1, 2))
    
    def Rot270(self, image_array): # rotate_270
        if image_array.ndim == 2:  # Single 2D image
            return np.rot90(image_array, k=3)
        else:  # Multi-channel image
            return np.rot90(image_array, k=3, axes=(1, 2))
    
    def VF_Rot90(self, image_array): # rotate_90_flipped_VF
        if image_array.ndim == 2:  # Single 2D image
            return np.rot90(np.flipud(image_array), k=1)
        else:
            return np.array([np.rot90(np.flipud(channel), k=1) for channel in image_array])

    def HF_Rot90(self, image_array): # rotate_90_flipped_HF
        if image_array.ndim == 2:  # Single 2D image
            return np.rot90(np.fliplr(image_array), k=1)
        else:
            return np.array([np.rot90(np.fliplr(channel), k=1) for channel in image_array])

    def augment_and_visualize(self, image_array, super_title=None):
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
        plt.figure(figsize=(12, 4))
        titles = ['Org', 'VF', 'HF', 'Rot90', 'Rot180', 'Rot270', 'VF+Rot90', 'HF+Rot90']
        for i, (title, aug_image) in enumerate(augments.items()):
            plt.subplot(1, 8, i+1)
            if aug_image.ndim == 3 and aug_image.shape[0] == 3:
                plt.imshow(aug_image.transpose(1, 2, 0), cmap='gray')
            else:
                plt.imshow(aug_image.squeeze(), cmap='gray')
            plt.title(titles[i])
            plt.axis('off')
            if super_title:
                plt.suptitle(super_title)
        plt.tight_layout()
        plt.show()

        # Visualize the results
        plt.figure(figsize=(12, 3*(len(augments)//3+1)))
        for i, (title, aug_image) in enumerate(augments.items()):
            plt.subplot(len(augments)//3+1, 3, i+1)
            if aug_image.ndim == 3 and aug_image.shape[0] == 3:
                plt.imshow(aug_image.transpose(1, 2, 0), cmap='gray')
            else:
                plt.imshow(aug_image.squeeze(), cmap='gray')
            plt.title(title)
            plt.axis('off')
            if super_title:
                plt.suptitle(super_title)
        plt.tight_layout()
        plt.show()
