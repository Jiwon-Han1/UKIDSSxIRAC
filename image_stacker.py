import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from PIL import Image

class ImageStacker:
    def __init__(self, save_dir, pop_channel=None, channels_order=None, model_name=None,
                 normalization=True, logarithm=False, standardization=False, zeroflux_sub=None,
                 replace_nan=False, replace_nan_val=None, zeroflux_sub_val=None,
                 channel_matching=None, marking=False, pre_processed=False, verbose=None):
        if pop_channel is None and channels_order is None:
            from config import channels_order as default_channels_order
            pop_channel = 'ch1'
            channels_order = default_channels_order
        elif pop_channel is not None and channels_order is None:
            raise(ValueError("'channels_order' should be specified if 'pop_channel' is given"))
        elif pop_channel is None and channels_order is not None:
            raise(ValueError("'pop_channel' should be specified if 'channels_order' is given"))
        if replace_nan and (replace_nan_val is None):
            raise (ValueError("'replace_nan_val' should be given if 'replace_nan' is True"))
        if zeroflux_sub and (zeroflux_sub_val is None):
            zeroflux_sub_val = 'median'
        if channel_matching is None:
            from config import match_output_channel_num as default_channel_matching
            channel_matching = default_channel_matching
        if verbose is None:
            from config import verbose_image_stacking as default_verbose_image_stacking
            verbose = default_verbose_image_stacking

        self.save_dir = save_dir
        self.pop_channel = pop_channel
        self.channels_order = channels_order
        self.normalization = normalization
        self.logarithm = logarithm
        self.standardization = standardization
        self.zeroflux_sub = zeroflux_sub
        self.replace_nan = replace_nan
        self.replace_nan_val = replace_nan_val
        self.zeroflux_sub_val = zeroflux_sub_val
        self.channel_matching = channel_matching
        self.marking = marking
        self.model_name = model_name
        self.pre_processed = pre_processed
        self.verbose = verbose

    def normalize_image(self, image):
        image = image.astype(np.float32)
        if np.isnan(image).any():
            min_val, max_val = np.nanmin(image), np.nanmax(image)
        else:
            min_val, max_val = np.min(image), np.max(image)
        if max_val > min_val:
            image = (image - min_val) / (max_val - min_val)
        return image

    def log_transform(self, image):
        if np.isnan(image).any():
            image = np.nan_to_num(image, nan=0.0)
        return np.log1p(image)  # log(1 + image) to avoid log(0) issues
    
    def standardize_image(self, image):
        image = image.astype(np.float32)
        if np.isnan(image).any():
            mean = np.nanmean(image)
            std = np.nanstd(image)
        else:
            mean = np.mean(image)
            std = np.std(image)
        if std > 0:
            image = (image - mean) / std
        return image
    
    def load_and_stack(self, target, target_datalist):
        channel_data = {}
        for target_data in target_datalist:
            band = os.path.splitext(os.path.split(target_data)[1].split(str(target))[1].split('_')[1])[0]
            nparray = np.load(target_data)

            if self.replace_nan:
                nparray[np.isnan(nparray)] = self.replace_nan_val
            if self.zeroflux_sub:
                if self.zeroflux_sub_val == 'median':
                    self.zeroflux_sub_val = np.nanmedian(nparray)
                elif self.zeroflux_sub_val == 'mean':
                    self.zeroflux_sub_val = np.nanmean(nparray)
                elif type(self.zeroflux_sub_val) is int:
                    pass
                nparray = nparray - self.zeroflux_sub_val
                nparray[np.where(nparray < 0.0)] = 0
            if self.normalization:
                nparray = self.normalize_image(nparray)
            if self.logarithm:
                nparray = self.log_transform(nparray)
            if self.standardization:
                nparray = self.standardize_image(nparray)
            channel_data[band] = nparray

        y_data = channel_data.pop(self.pop_channel)
        X_data = np.stack([channel_data[ch] for ch in self.channels_order], axis=0)
        if self.channel_matching:
            y_data = np.repeat(y_data[np.newaxis, :, :], len(self.channels_order), axis=0)

        return X_data, y_data   # X_data.shape: (C, H, W)
    
    def mark_on_name(self):
        pre_processed =  ''
        if self.marking:
            if self.normalization:
                pre_processed = pre_processed + '_norm'
            if self.logarithm:
                pre_processed = pre_processed + '_log'
            if self.standardization:
                pre_processed = pre_processed + '_std'
            if self.zeroflux_sub:
                pre_processed = pre_processed + '_zeroflux_sub'
        return pre_processed
    
    def save_data_for_cyclegan(self):
        if self.model_name == 'cycle_gan_junyanz':
            return 'A', 'B'
        else:
            return '', ''
    
    def save_name(self, save_dir, target, ext, aug=''):
        pre_processed = self.mark_on_name()
        cyclegan_dir = self.save_data_for_cyclegan()
        filenames = [os.path.join(save_dir+cyclegan_dir[i],f'{target}_{suffix}{pre_processed}{aug}.{ext}') 
                     for i, suffix in enumerate(['X', 'y'])]
        return filenames
        
    def save_as_npy(self, target, target_datalist, save_dir=None, verbose=False):
        X_data, y_data = self.load_and_stack(target, target_datalist)
        save_dir = self.save_dir if save_dir is None else save_dir

        filename_X, filename_y = self.save_name(save_dir, target, 'npy')

        if verbose:
            print('\n'+'[[ # '+target+' ]]')
            print('X: ', np.shape(X_data), filename_X)
            print('y: ', np.shape(y_data), filename_y)

        np.save(filename_X, X_data)
        np.save(filename_y, y_data)

    def save_as_png(self, target, target_datalist, save_dir=None, verbose=False):
        X_data, y_data = self.load_and_stack(target, target_datalist)
        save_dir = self.save_dir if save_dir is None else save_dir

        filename_X, filename_y = self.save_name(save_dir, target, 'png')

        if verbose:
            print('\n'+'[[ # '+target+' ]]')
            print('X: ', np.shape(X_data), filename_X)
            print('y: ', np.shape(y_data), filename_y)

        if self.normalization: # self.normalization=False일 때 확인 필요
            X_data, y_data =  (X_data * 255).astype(np.uint8), (y_data * 255).astype(np.uint8)

        X_img = np.stack([X_data[i] for i in range(3)], axis=-1) # RGB 스택 순서 맞는지 확인 필요
        X_img = Image.fromarray(X_img)
        y_img = Image.fromarray(y_data, mode='L') # self.channel_matching=True일 때 대비 변경 필요

        X_img.save(filename_X)
        y_img.save(filename_y)

    def save_as_jpg(self, target, target_datalist, save_dir=None, verbose=False):
        X_data, y_data = self.load_and_stack(target, target_datalist)
        save_dir = self.save_dir if save_dir is None else save_dir

        filename_X, filename_y = self.save_name(save_dir, target, 'jpg')

        if verbose:
            print('\n'+'[[ # '+target+' ]]')
            print('X: ', np.shape(X_data), filename_X)
            print('y: ', np.shape(y_data), filename_y)

        if self.normalization: # self.normalization=False일 때 확인 필요
            X_data, y_data = (X_data * 255).astype(np.uint8), (y_data * 255).astype(np.uint8)
            print(np.min(X_data), np.max(X_data))

        # 데이터 확인을 위한 중간 출력
        print(f"X_data min: {np.min(X_data)}, max: {np.max(X_data)}")
        print(f"y_data min: {np.min(y_data)}, max: {np.max(y_data)}")
        
        plt.figure(figsize = (8, 3))
        plt.subplot(1, 2, 1)
        plt.imshow(X_data.transpose(1, 2, 0), cmap='gray', vmin=np.min(X_data), vmax=np.max(X_data))
        plt.axis('off')
        plt.title(f'{target}_X')

        plt.subplot(1, 2, 2)
        plt.imshow(y_data, cmap='gray')
        plt.axis('off')
        plt.title(f'{target}_y')

        print(np.shape(X_data.transpose(1, 2, 0)))

        cv2.imwrite(filename_X, X_data.transpose(1, 2, 0))
        cv2.imwrite(filename_y, y_data) # self.channel_matching=True일 때 대비 변경 필요
