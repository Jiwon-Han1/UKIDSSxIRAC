import numpy as np
import os, copy
from astropy.io import fits
import cv2
import reproject

class FitsArrayConverter:
    def __init__(self, filename, stacking_size=None, dir_name=None):
        if stacking_size is None:
            from config import stacking_size as default_stacking_size
            stacking_size = default_stacking_size
        if dir_name is None:
            dir_name = 'data'

        self.filename = filename
        self.stacking_size = stacking_size
        self.dir_name = dir_name

    def get_image(self):
        file = fits.open(self.filename)
        # file.info() ###
        image = copy.deepcopy(file[0].data)     # type(image): numpy.ndarray (UKIDSS: float64 / IRAC: f4)
        return image

    def resize(self, org_array, edit_pix):
        if edit_pix < org_array.shape[0]:
            interpolation = cv2.INTER_AREA
        else:
            interpolation = cv2.INTER_LINEAR
        return cv2.resize(org_array, (edit_pix, edit_pix), interpolation=interpolation)
    
    def upsample(self, org_array, num):
        return org_array.repeat(num, axis=0).repeat(num, axis=1)

    def select_frame(self, fits_filename, verbose=None):
        if verbose is None:
            from config import verbose_fits_info as default_verbose_fits_info
            verbose = default_verbose_fits_info

        file = fits.open(fits_filename)
        if verbose:
            file.info()
        if len(file) > 1:
            file.info()
            imagenum = input("Insert the frame number on the FITS info and press the [Enter] button.")
            if imagenum > len(file)-1:
                raise ValueError(f"Frame number should be smaller than {len(file)-1}!")
        else:
            imagenum = 0
        return file, imagenum

    def projection_resize(self, ref_header_filename):
        ref_file, ref_num = self.select_frame(ref_header_filename)
        data_file, data_num = self.select_frame(self.filename)
        ref_header = ref_file[ref_num].header
        image_reprojected, _ = reproject.reproject_interp(data_file[data_num], ref_header)
        return image_reprojected
        
    def save_as_npy(self, image, pre_processed=''):
        from config import root_dir
        savename = root_dir+'/'+self.dir_name+'/'+self.filename.split('images/')[1]
        # print(os.path.splitext(savename)[0]+pre_processed+'.npy')
        np.save(os.path.splitext(savename)[0]+pre_processed+'.npy', image)
    
    def save_image(self, org_preserve=None, resize=None, size_pix=None, upsample=None, 
                   registration=None, ref_header_filename=None, match_size=False, marking=False, verbose=None):
        if size_pix is None:
            from config import stacking_size
            size_pix=stacking_size  #self.stacking_size
        if (registration is True) and (ref_header_filename is None):
            raise ValueError("The 'ref_header_filename' must be specified when 'registration' is True.")
        if not (org_preserve or resize or upsample or registration):
            raise ValueError("One of the following options must be set to True: 'resize', 'upsample', or 'registration'.")
        if verbose is None:
            from config import verbose_fits_to_npy as default_verbose_fits_to_npy
            verbose = default_verbose_fits_to_npy
        
        image = self.get_image()
        org_size = np.shape(image)

        pre_processed = ''
        if org_preserve:
            pre_processed = pre_processed+f'_org'
        if resize:
            image = self.resize(image, size_pix)
            pre_processed = pre_processed+f'_resized_{size_pix}x{size_pix}'
        if upsample: # IRAC images should be upsampled
            up_num = round(self.stacking_size/size_pix)
            if up_num > 1:
                image = self.upsample(image, up_num)
                pre_processed = pre_processed+f'_upsampled_{up_num}X'
            else:
                pre_processed = pre_processed+f'_resized_{size_pix}x{size_pix}'
        if registration:
            image = self.projection_resize(ref_header_filename)
            pre_processed = pre_processed+f'_aligned'
            projected_shape = image.shape
            if match_size:
                current_shape = image.shape
                target_shape = (size_pix, size_pix)
                if current_shape[0] > target_shape[0]:
                    extra_height = (current_shape[0] - target_shape[0]) // 2
                    image = image[extra_height:extra_height + target_shape[0], :]
                    current_shape = np.shape(image)
                if current_shape[1] > target_shape[1]:
                    extra_width = (current_shape[1] - target_shape[1]) // 2
                    image = image[:, extra_width:extra_width + target_shape[1]]
                    current_shape = np.shape(image)
                if current_shape != target_shape:
                    padded_image = np.full(target_shape, np.nan, dtype=image.dtype)
                    min_row = (target_shape[0] - current_shape[0]) // 2
                    min_col = (target_shape[1] - current_shape[1]) // 2
                    padded_image[min_row:min_row + current_shape[0], min_col:min_col + current_shape[1]] = image
                    image = padded_image
                pre_processed = pre_processed + f'_nan_filled'
        if not marking:
            pre_processed = ''
        if verbose:
            band = os.path.splitext(self.filename)[0].split('_')[-1]
            if registration and match_size:
                print(f'{band}:   {org_size}  ==>  {projected_shape}  ==>  {np.shape(image)}')
            else:
                print(f'{band}:   {org_size}  ==>  {np.shape(image)}')

        self.save_as_npy(image, pre_processed)
