class FitsRegistrationManager:
    def __init__(self):
        pass
    
    def select_frame(self, fits_filename, verbose=None):
        if verbose is None:
            from config import verbose_fits_info as default_verbose_fits_info
            verbose = default_verbose_fits_info
        file = fits.open(fits_filename)
        if verbose:
            file.info()
        if len(file) > 2:
            file.info()
            imagenum = input("Insert the frame number on the FITS info and press the [Enter] button.")
            if imagenum > len(file)-1:
                raise ValueError(f"Frame number should be smaller than {len(file)-1}!")
        else:
            imagenum = -1
        return file, imagenum

    def resize_with_projection(self, filename, ref_header_filename):
        '''
        Parameters:
        - reprojected_image: 2D numpy array of the reprojected image.
        - org_data_header: FITS header from the original image being reprojected (IRAC header).
        - ref_header: FITS header from the reference image (VVV header).
        '''
        ref_file, ref_num = self.select_frame(ref_header_filename)
        data_file, data_num = self.select_frame(filename)
        ref_header, org_data_header = ref_file[ref_num].header, data_file[data_num].header
        reprojected_image, _ = reproject.reproject_interp(data_file[data_num], ref_header)
        return reprojected_image, org_data_header, ref_header
    
    def update_header(self, reprojected_image, org_data_header, ref_header, update_keys=None, verbose=False):
        if verbose:
            ref_header
        if update_keys is None:
            update_keys = ['CTYPE1', 'CTYPE2', 'CRVAL1', 'CRVAL2', 'CRPIX1', 'CRPIX2', 'CD1_1', 'CD1_2', 'CD2_1', 'CD2_2']
        
        # Update information to match the reprojected image
        new_header = copy.deepcopy(org_data_header)
        for key in update_keys:
            if key in ref_header:
                new_header[key] = ref_header[key]
        new_header['NAXIS'] = 2
        new_header['NAXIS1'] = reprojected_image.shape[1]  # Width of the image
        new_header['NAXIS2'] = reprojected_image.shape[0]  # Height of the image
        new_header['COMMENT'] = "Reprojected to match the VVV coordinate system."
        new_header['HISTORY'] = "Image reprojected using astropy.reproject."
        return new_header
    
    def save_as_npy(self, reprojected_image, save_name, verbose=False):#, filename, ref_header_filename):
        if verbose:
            print(f'Reprojected Image Size: {reprojected_image.shape}')
            print(f'File Saved as {os.path.basename(save_name)}')
        np.save(save_name, reprojected_image)

    def save_as_fits(self, reprojected_image, new_header, save_name, verbose=False):
        if verbose:
            print(f'Reprojected Image Size: {reprojected_image.shape}')
            print(f'File Saved as {os.path.basename(save_name)}')
        hdu = fits.PrimaryHDU(data=reprojected_image, header=new_header)
        hdu_list = fits.HDUList([hdu])
        hdu_list.writeto(save_name, overwrite=True)

    def get_image(self, filename, header=False):
        file = fits.open(filename)
        image = file[-1].data
        if header:
            header = file[-1].header
            return image, header
        else:
            return image
        
    def get_vmin_vmax(self, image, lower_percentile=1, upper_percentile=99):
        vmin = np.nanpercentile(image, lower_percentile)
        vmax = np.nanpercentile(image, upper_percentile)
        return vmin, vmax
 
    def visualize_result(self, filename, ref_header_filename):
        ref_image, ref_header = self.get_image(ref_header_filename, header=True)
        reprojected_image = self.resize_with_projection(filename, ref_header_filename)[0]
        fig = plt.figure(figsize = (22, 10))

        ax1 = plt.subplot(121, projection=WCS(ref_header))
        min_val, max_val = self.get_vmin_vmax(ref_image, lower_percentile=2, upper_percentile=98)
        ax1.imshow(ref_image, cmap='cividis', origin='lower', vmax=max_val, vmin=min_val)
        ax1.set_xlabel(r'RA')
        ax1.set_ylabel(r'Dec')
        ax1.grid(color='white', ls='dotted')
        _, title = os.path.basename(ref_header_filename.split('.')[0]).split('gc_')
        title = title.split('_')[0:2]
        ax1.set_title(f'Ref Coord: {title[0].upper()}_{title[1]}')

        ax2 =  plt.subplot(122, projection=WCS(ref_header))
        min_val, max_val = self.get_vmin_vmax(reprojected_image, lower_percentile=2, upper_percentile=98)
        ax2.imshow(reprojected_image, cmap='cividis', origin='lower', vmax=max_val, vmin=min_val)
        ax2.set_xlabel(r'RA')
        ax2.set_ylabel(r'Dec')
        ax2.grid(color='white', ls='dotted')
        _, title = os.path.basename(filename.split('.')[0]).split('gc_')
        title = title.split('_')[0:2]
        ax2.set_title(f'Ref Coord: {title[0].upper()}_{title[1]}')

    def conduct_registration(self, filename, ref_header_filename, save_name, save_as_fits=True, save_as_npy=False, verbose=False):
        '''
        Conducts the full registration process: load, reproject, update header, and save.
    
        Parameters:
        - filename: str, path to the image file to be reprojected.
        - ref_header_filename: str, path to the reference header file.
        - save_name: str, output file name for saving the reprojected image.
        - save_as_fits: bool, whether to save the result as a FITS file (default: True).
        - save_as_npy: bool, whether to save the result as a NumPy file (default: False).
        - verbose: bool, whether to print detailed logs.
        '''
        if verbose:
            print(f"Starting registration for: {os.path.basename(filename)}")
            print(f"Reference header: {os.path.basename(ref_header_filename)}")
            print(f"Save as FITS: {save_as_fits}, Save as NumPy: {save_as_npy}")

        # Step 1: Reproject the image
        reprojected_image, org_data_header, ref_header = self.resize_with_projection(filename, ref_header_filename)
        if verbose:
            print("1. Reprojection completed.")

        # Step 2: Update the header
        new_header = self.update_header(reprojected_image, org_data_header, ref_header, verbose=verbose)
        if verbose:
            print("2. Header updated.")

        # Step 3: Save the reprojected image
        if save_as_fits:
            fits_save_name = f"{save_name}.fits"
            self.save_as_fits(reprojected_image, new_header, fits_save_name, verbose=verbose)
            if verbose:
                print(f"3. FITS file saved: {fits_save_name}")
        if save_as_npy:
            npy_save_name = f"{save_name}.npy"
            self.save_as_npy(reprojected_image, npy_save_name, verbose=verbose)
            if verbose:
                print(f"3. NumPy file saved: {npy_save_name}")

        if verbose:
            print("#-- Registration process completed.")
