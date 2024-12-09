import numpy as np
import matplotlib.pyplot as plt
import os, copy
from astropy.io import fits
from astropy.wcs import WCS

# Projection Result Visualizer
class ProjectionVisualizer:
    def __init__(self):
        pass

    def get_vmin_vmax(self, image, lower_percentile=1, upper_percentile=99):
        vmin = np.nanpercentile(image, lower_percentile)
        vmax = np.nanpercentile(image, upper_percentile)
        return vmin, vmax
    
    def save_image(self, ref_data_filename, data_filename, ref_header, targetname=None, 
                   lower_percentile=None, upper_percentile=None, save_image_dir=None):
        if lower_percentile is None:
            lower_percentile = 0
        if upper_percentile is None:
            upper_percentile = 100
        if save_image_dir is None:
            from config import save_dir
            save_image_dir = os.path.join(os.path.split(save_dir)[0],'visualize')
            if not os.path.exists(save_image_dir):
                print(f'\n=== Making "data/{os.path.split(os.path.split(save_image_dir)[0])[-1]}/visualize" Directory ===\n')
                os.makedirs(save_image_dir)

        image_ref = np.load(ref_data_filename)
        image = np.load(data_filename)
        fig = plt.figure(figsize = (22, 10))

        # UKIDSS - K Band (Ref Coord.)
        ax1 = plt.subplot(121, projection=WCS(ref_header))
        min_val, max_val = self.get_vmin_vmax(image_ref, lower_percentile=lower_percentile, upper_percentile=upper_percentile)
        ax1.imshow(image_ref, cmap='cividis', origin='lower', vmax=max_val, vmin=min_val)
        ax1.set_xlabel(r'RA')
        ax1.set_ylabel(r'Dec')
        ax1.grid(color='white', ls='dotted')

        # Aligned Data
        ax2 =  plt.subplot(122, projection=WCS(ref_header))
        min_val, max_val = self.get_vmin_vmax(image, lower_percentile=lower_percentile, upper_percentile=upper_percentile)
        ax2.imshow(image, cmap='cividis', origin='lower', vmax=max_val, vmin=min_val)
        ax2.set_xlabel(r'RA')
        ax2.set_ylabel(r'Dec')
        ax2.grid(color='white', ls='dotted')

        _, title1 = os.path.splitext(ref_data_filename)[0].split(str(targetname))
        title1 = title1.split('_')[1]
        ax1.set_title(f'Ref Coord: SSTGC {targetname}_{title1}')

        _, title2 = os.path.splitext(data_filename)[0].split(str(targetname))
        ax2.set_title(f'Aligned Result: SSTGC {targetname}{title2}')

        save_name = os.path.join(save_image_dir, f'{targetname}{title2}.png')
        plt.savefig(save_name, dpi=300)
        print(f'\n=== File saved successfully: {save_name} ===\n')
        plt.close(fig)


# Visualize Inference Result 
'''
Real A - Fake B - Reconstructed A
Real B - Fake A - Reconstructed B
'''
class ResultVisualizer:
    def __init__(self, test_target_name, test_real_A, fake_B, recovered_A, 
                      test_real_B, fake_A, recovered_B, channels_order=None, channel='K', target_channel=None,
                      match_output_channel_num=None):
        self.test_target_name = test_target_name
        self.test_real_A = test_real_A
        self.fake_B = fake_B
        self.recovered_A = recovered_A
        self.test_real_B = test_real_B
        self.fake_A = fake_A
        self.recovered_B = recovered_B
        if channels_order is None:
            from config import channels_order as default_channels_order
            self.channels_order = default_channels_order
        else:
            self.channels_order = channels_order
        self.channel = channel
        
        if self.channel in self.channels_order:
            self.ch_idx = self.channels_order.index(self.channel)
        else:
            raise ValueError("Please check the channel name.")
        
        if target_channel is None:
            target_channel = 'ch1'
        self.target_channel = target_channel

        if match_output_channel_num is None:
            from config import match_output_channel_num as default_match_output_channel_num
            match_output_channel_num = default_match_output_channel_num
            
        if match_output_channel_num:
            self.images = [self.test_real_A[0, self.ch_idx, :, :].cpu().numpy(), 
                           self.fake_B[0, self.ch_idx, :, :].cpu().numpy(), 
                           self.recovered_A[0, self.ch_idx, :, :].cpu().numpy(),
                           self.test_real_B[0, self.ch_idx, :, :].cpu().numpy(), 
                           self.fake_A[0, self.ch_idx, :, :].cpu().numpy(), 
                           self.recovered_B[0, self.ch_idx, :, :].cpu().numpy()]
        else:
            self.images = [self.test_real_A[0, self.ch_idx, :, :].cpu().numpy(), 
                           self.fake_B[0, 0, :, :].cpu().numpy(),  # Adjust for 1-channel output
                           self.recovered_A[0, self.ch_idx, :, :].cpu().numpy(),
                           self.test_real_B[0, 0, :, :].cpu().numpy(),  # Adjust for 1-channel output
                           self.fake_A[0, self.ch_idx, :, :].cpu().numpy(), 
                           self.recovered_B[0, 0, :, :].cpu().numpy()]  # Adjust for 1-channel output
        self.titles = ['Real A', 'Fake B', 'Reconstructed A', 'Real B', 'Fake A', 'Reconstructed B']
    
    def get_vmin_vmax(self, image, lower_percentile=1, upper_percentile=99):
        vmin = np.percentile(image, lower_percentile)
        vmax = np.percentile(image, upper_percentile)
        return vmin, vmax

    def result_plot(self, show_colorbar=False, set_color_scale=False, percentile_range=(1,99)):
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        for i, ax in enumerate(axs.flat):
            if set_color_scale:
                vmin, vmax = self.get_vmin_vmax(self.images[i], percentile_range[0], percentile_range[1])
                im = ax.imshow(self.images[i], vmin=vmin, vmax=vmax)
            else:
                im = ax.imshow(self.images[i])#, cmap='gray')
            ax.set_title(self.titles[i])
            ax.axis('off')
            if show_colorbar:
                fig.colorbar(im, ax=ax, orientation='horizontal')

        from config import filters1, filters2
        if self.target_channel in filters1:
            title_target_channel = f'UKIDSS {self.target_channel} band'
        elif self.target_channel in filters2:
            title_target_channel = f'IRAC {self.target_channel}'
        if self.channel in filters1:
            suptitle_suffix = f'UKIDSS {self.channel} band <--> {title_target_channel}'
        elif self.channel in filters2:
            suptitle_suffix = f'IRAC {self.channel} <-> {title_target_channel}'
        plt.suptitle(f'SSTGC {self.test_target_name[0]}: {suptitle_suffix}', fontsize='xx-large')
        plt.tight_layout()
        plt.show()

    def result_histogram(self, set_color_scale=False, percentile_range=(1,99)):
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        for i, ax in enumerate(axs.flat):
            ax.hist(self.images[i].ravel(), bins=256, histtype='step', color='black')
            ax.set_xlim(self.get_vmin_vmax(self.images[i], lower_percentile=0, upper_percentile=99))
            ax.set_title(f'{self.titles[i]} Histogram')
            if set_color_scale:
                vmin, vmax = self.get_vmin_vmax(self.images[i], percentile_range[0], percentile_range[1])
                ylim = ax.get_ylim()[1]
                ax.plot([vmin,vmin],[0,ylim], 'r--')
                ax.plot([vmax,vmax],[0,ylim], 'r--')
                ax.set_ylim([0,ylim])

        plt.suptitle(f'SSTGC {self.test_target_name[0]}', fontsize='xx-large')
        plt.tight_layout()
        plt.show()
    
    def result_evaluate(self):
        pass
