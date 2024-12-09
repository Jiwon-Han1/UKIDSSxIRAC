# utils/utils.py
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pickle
import json
from dataset import CustomDataset

# Weight Initialize - Params were empirically decided
def weights_init_normal(m): # In-Place Method
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

# Linearly Decaying Learning Rate
class LambdaLR():
    def __init__(self, n_epochs, offset, decay_start_epoch):
        assert ((n_epochs - decay_start_epoch) > 0), "Decay must start before the training session ends!"
        self.n_epochs = n_epochs
        self.offset = offset
        self.decay_start_epoch = decay_start_epoch

    def step(self, epoch):
        return 1.0 - max(0, epoch + self.offset - self.decay_start_epoch)/(self.n_epochs - self.decay_start_epoch)
    
# Save Checkpoint
def save_checkpoint(epoch, model_G_A2B, model_G_B2A, model_D_A, model_D_B, 
                    optimizer_G, optimizer_D_A, optimizer_D_B, scheduler_G, scheduler_D_A, scheduler_D_B,
                    best_val_loss, best_epoch, stacking_size=None, input_nc=None, param_dir=None, 
                    status='best', pre_processed='None', aug=None, add_suffix=''):
    
    if stacking_size is None:
        from config import stacking_size as default_stacking_size
        stacking_size = default_stacking_size
    if input_nc is None:
        from config import input_nc as default_input_nc
        input_nc = default_input_nc
    if param_dir is None:
        from config import param_dir as default_param_dir
        param_dir = default_param_dir
    aug = '_augmented' if aug is not None else ''
    if pre_processed is None:
        raise ValueError('pre_processed must be specified')
    
    file_suffix = f'{aug}_{stacking_size}x{stacking_size}_{input_nc}_channels{add_suffix}{pre_processed}.pth'
    filename = f'{param_dir}CycleGAN_checkpoint_{status}'+file_suffix
    torch.save({
        'epoch': epoch,
        'model_G_A2B_state_dict': model_G_A2B.state_dict(),
        'model_G_B2A_state_dict': model_G_B2A.state_dict(),
        'model_D_A_state_dict': model_D_A.state_dict(),
        'model_D_B_state_dict': model_D_B.state_dict(),
        'optimizer_G_state_dict': optimizer_G.state_dict(),
        'optimizer_D_A_state_dict': optimizer_D_A.state_dict(),
        'optimizer_D_B_state_dict': optimizer_D_B.state_dict(),
        'scheduler_G_state_dict': scheduler_G.state_dict(),
        'scheduler_D_A_state_dict': scheduler_D_A.state_dict(),
        'scheduler_D_B_state_dict': scheduler_D_B.state_dict(),
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch
        }, filename)
    print(f'Saved as {filename}')

# Load Checkpoint
def load_checkpoint(netG_A2B, netG_B2A, netD_A, netD_B, 
                    optimizer_G, optimizer_D_A, optimizer_D_B, scheduler_G, scheduler_D_A, scheduler_D_B,
                    stacking_size=None, input_nc=None, param_dir=None, 
                    status='best', pre_processed='None', aug=None, add_suffix=''):
    if stacking_size is None:
        from config import stacking_size as default_stacking_size
        stacking_size = default_stacking_size
    if input_nc is None:
        from config import input_nc as default_input_nc
        input_nc = default_input_nc
    if param_dir is None:
        from config import param_dir as default_param_dir
        param_dir = default_param_dir
    aug = '_augmented' if aug is not None else ''
    if pre_processed is None:
        raise ValueError('pre_processed must be specified')

    file_suffix = f'{aug}_{stacking_size}x{stacking_size}_{input_nc}_channels{add_suffix}{pre_processed}.pth'
    filename = f'{param_dir}CycleGAN_checkpoint_{status}'+file_suffix
    checkpoint = torch.load(filename)

    start_epoch = checkpoint['epoch'] + 1
    netG_A2B.load_state_dict(checkpoint['model_G_A2B_state_dict'])
    netG_B2A.load_state_dict(checkpoint['model_G_B2A_state_dict'])
    netD_A.load_state_dict(checkpoint['model_D_A_state_dict'])
    netD_B.load_state_dict(checkpoint['model_D_B_state_dict'])
    optimizer_G.load_state_dict(checkpoint['optimizer_G_state_dict'])
    optimizer_D_A.load_state_dict(checkpoint['optimizer_D_A_state_dict'])
    optimizer_D_B.load_state_dict(checkpoint['optimizer_D_B_state_dict'])
    scheduler_G.load_state_dict(checkpoint['scheduler_G_state_dict'])
    scheduler_D_A.load_state_dict(checkpoint['scheduler_D_A_state_dict'])
    scheduler_D_B.load_state_dict(checkpoint['scheduler_D_B_state_dict'])
    best_val_loss = checkpoint['best_val_loss']
    best_epoch = checkpoint['best_epoch']
    return start_epoch, best_val_loss, best_epoch

# Load Checkpoint for Inference
def load_checkpoint_for_test(netG_A2B, netG_B2A, stacking_size=None, input_nc=None,
                             param_dir=None, status='best', pre_processed='None', aug=None, add_suffix=''):
    
    if stacking_size is None:
        from config import stacking_size as default_stacking_size
        stacking_size = default_stacking_size
    if input_nc is None:
        from config import input_nc as default_input_nc
        input_nc = default_input_nc
    if param_dir is None:
        from config import param_dir as default_param_dir
        param_dir = default_param_dir
    aug = '_augmented' if aug is not None else ''
    if pre_processed is None:
        raise ValueError('pre_processed must be specified')
    
    file_suffix = f'{aug}_{stacking_size}x{stacking_size}_{input_nc}_channels{add_suffix}{pre_processed}.pth'
    filename = f'{param_dir}CycleGAN_checkpoint_{status}'+file_suffix
    checkpoint = torch.load(filename)

    netG_A2B.load_state_dict(checkpoint['model_G_A2B_state_dict'])
    netG_B2A.load_state_dict(checkpoint['model_G_B2A_state_dict'])
    print(f'Loaded checkpoint from: \n{filename}')
    return filename

# Keep Newly Generated 50 Images
def update_buffer(buffer, data, max_size=50):
    buffer.append(data)
    if len(buffer) > max_size:
        buffer.pop(0)
    return buffer

# Gather Old & New Image
def sample_buffer(buffer, data):
    if len(buffer) < 50:
        buffer.append(data)
        return data
    if np.random.rand() > 0.5:
        return buffer[np.random.randint(len(buffer))]
    else:
        buffer[np.random.randint(len(buffer))] = data
        return data
    
# Memory Alloscation
def allocate_memory(batch_size=None, input_nc=None, output_nc=None,
                    stacking_size=None, netD_A=None, device='cuda'):
    if batch_size is None:
        from config import batch_size as default_batch_size
        batch_size = default_batch_size
    if input_nc is None:
        from config import input_nc as default_input_nc
        input_nc = default_input_nc
    if output_nc is None:
        from config import output_nc as default_output_nc
        output_nc = default_output_nc
    if stacking_size is None:
        from config import stacking_size as default_stacking_size
        stacking_size = default_stacking_size
    if netD_A is None:
        raise ValueError("netD_A must be provided")
    
    # Initialize input tensors
    input_A = torch.randn(batch_size, input_nc, stacking_size, stacking_size, device=device, requires_grad=False)
    input_B = torch.randn(batch_size, output_nc, stacking_size, stacking_size, device=device, requires_grad=False)

    # Initialize with correct shapes
    with torch.no_grad():
        sample_output = netD_A(input_A)
        output_shape = sample_output.shape[2:]  # Get H, W from the sample output

    target_real = torch.ones(batch_size, 1, *output_shape, device=device, requires_grad=False)
    target_fake = torch.zeros(batch_size, 1, *output_shape, device=device, requires_grad=False)

    return input_A, input_B, target_real, target_fake

# Check the Batch
def extract_first_batch(type='val', visualize_num=2, saved_dir=None):
    if saved_dir is None:
        from config import dataset_dir
        saved_dir = dataset_dir
    with open(f'{saved_dir}/{type}_data.pkl', 'rb') as f:
        data = pickle.load(f)
    from config import transform
    dataset = CustomDataset(data, transform=transform)
    dataset_iter = iter(dataset)

    # Check the Shape
    batch_first = next(dataset_iter)
    print(f'Image Shape (X): {batch_first[0].shape}')
    print(f'Image Shape (y): {batch_first[1].shape}')

    # Visualize Dataset
    plt.figure(figsize=(8, 4*visualize_num))
    for i in range(visualize_num):
        image_X, image_y = next(dataset_iter)
        image_X = image_X.permute(1, 2, 0).numpy()
        image_y = image_y.permute(1, 2, 0).numpy()

        plt.subplot(visualize_num, 2, 2*i+1)
        plt.imshow(image_X[:,:,2])#, cmap='gray')
        plt.title(f'UKIDSS - JBand')
        plt.axis('off')

        plt.subplot(visualize_num, 2, 2*i+2)
        plt.imshow(image_y[:,:,0])#, cmap='gray')
        plt.title(f'IRAC - Ch.1')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

# Save and Load Loss
class LossManager:
    def __init__(self, save_dir=None, stacking_size=None, input_nc=None, pre_processed=None, aug=None, add_suffix=''):
        if input_nc is None:
            from config import input_nc as default_input_nc
            input_nc = default_input_nc
        if save_dir is None:
            from config import root_dir as default_save_dir
            save_dir = default_save_dir
        if stacking_size is None:
            from config import stacking_size as default_stacking_size
            stacking_size = default_stacking_size
        self.aug = '_augmented' if aug is not None else ''
        if pre_processed is None:
            raise ValueError('pre_processed must be specified')

        self.input_nc = input_nc
        self.save_dir = save_dir
        self.stacking_size = stacking_size
        self.pre_processed = pre_processed
        self.add_suffix = add_suffix

    def loss_filename(self):
        file_suffix = f'{self.aug}_{self.stacking_size}x{self.stacking_size}_{self.input_nc}_channels{self.add_suffix}{self.pre_processed}.json'
        loss_filename = f'{self.save_dir}/result/losses{file_suffix}'
        return loss_filename

    def save_loss(self, G_losses, D_A_losses, D_B_losses, val_G_losses, val_D_A_losses, val_D_B_losses):
        losses = {
            'G_losses': G_losses,
            'D_A_losses': D_A_losses,
            'D_B_losses': D_B_losses,
            'val_G_losses': val_G_losses,
            'val_D_A_losses': val_D_A_losses,
            'val_D_B_losses': val_D_B_losses
        }
        loss_filename = self.loss_filename()
        with open(loss_filename, 'w') as f:
            json.dump(losses, f)
        print(f"Loss Saved: {loss_filename}")

    def load_loss(self, loss_filename=None):
        if loss_filename is None:
            loss_filename = self.loss_filename()
        with open(loss_filename, 'r') as f:
            losses = json.load(f)
        print(f"Loss Loaded: {loss_filename}")
        return losses

    def visualize_loss_curve(self, figsize=None, losses=None, cut_epoch=None):
        if figsize is None:
            figsize = (14, 6)
        if losses is None:
            losses = self.load_loss()
        if cut_epoch is not None:
            from config import max_epochs
            loss_len = min([len(v) for v in losses.values()])
            if cut_epoch < min([max_epochs, loss_len]):
                losses = {k: v[:cut_epoch] for k, v in losses.items()}
            else:
                raise ValueError(f"cut_epoch ({cut_epoch}) is greater than loss length ({min([max_epochs, loss_len])}).")

        plt.figure(figsize=figsize)

        plt.subplot(1, 3, 1)
        plt.plot(losses['G_losses'], label='Generator Loss (Train)', color='blue')
        plt.plot(losses['val_G_losses'], label='Generator Loss (Val)', color='red')
        plt.title('Generator Losses')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')

        plt.subplot(1, 3, 2)
        plt.plot(losses['D_A_losses'], label='Discriminator_A Loss (Train)', color='blue')
        plt.plot(losses['val_D_A_losses'], label='Discriminator_A Loss (Val)', color='red')
        plt.title('Discriminator_A Losses')
        plt.xlabel('Epochs')
        # plt.ylabel('Loss')
        plt.legend(loc='upper right')
        
        plt.subplot(1, 3, 3)
        plt.plot(losses['D_B_losses'], label='Discriminator_B Loss (Train)', color='blue')
        plt.plot(losses['val_D_B_losses'], label='Discriminator_B Loss (Val)', color='red')
        plt.title('Discriminator_B Losses')
        plt.xlabel('Epochs')
        # plt.ylabel('Loss')
        plt.legend(loc='upper right')

        if cut_epoch is not None:
            for i in range(3):
                plt.subplot(1, 3, i+1)
                plt.xlim([0,cut_epoch-1])
                
        plt.tight_layout()
        plt.show()













# Save and Load Loss - for Generator Seperate Case
class GenSepLossManager:
    def __init__(self, save_dir=None, stacking_size=None, input_nc=None, pre_processed=None, aug=None, add_suffix=''):
        if input_nc is None:
            from config import input_nc as default_input_nc
            input_nc = default_input_nc
        if save_dir is None:
            from config import root_dir as default_save_dir
            save_dir = default_save_dir
        if stacking_size is None:
            from config import stacking_size as default_stacking_size
            stacking_size = default_stacking_size
        self.aug = '_augmented' if aug is not None else ''
        if pre_processed is None:
            raise ValueError('pre_processed must be specified')

        self.input_nc = input_nc
        self.save_dir = save_dir
        self.stacking_size = stacking_size
        self.pre_processed = pre_processed
        self.add_suffix = add_suffix

    def loss_filename(self):
        file_suffix = f'{self.aug}_{self.stacking_size}x{self.stacking_size}_{self.input_nc}_channels{self.add_suffix}{self.pre_processed}.json'
        loss_filename = f'{self.save_dir}/result/losses{file_suffix}'
        return loss_filename

    def save_loss(self, G_A2B_losses, G_B2A_losses, D_A_losses, D_B_losses, 
                  val_G_A2B_losses, val_G_B2A_losses, val_D_A_losses, val_D_B_losses):
        losses = {
            'G_A2B_losses': G_A2B_losses,
            'G_B2A_losses': G_B2A_losses,
            'D_A_losses': D_A_losses,
            'D_B_losses': D_B_losses,
            'val_G_A2B_losses': val_G_A2B_losses,
            'val_G_B2A_losses': val_G_B2A_losses,
            'val_D_A_losses': val_D_A_losses,
            'val_D_B_losses': val_D_B_losses
        }
        loss_filename = self.loss_filename()
        with open(loss_filename, 'w') as f:
            json.dump(losses, f)
        print(f"Loss Saved: {loss_filename}")

    def load_loss(self, loss_filename=None):
        if loss_filename is None:
            loss_filename = self.loss_filename()
        with open(loss_filename, 'r') as f:
            losses = json.load(f)
        print(f"Loss Loaded: {loss_filename}")
        return losses

    def visualize_loss_curve(self, figsize=None, losses=None, cut_epoch=None):
        if figsize is None:
            figsize = (14, 4)
        if losses is None:
            losses = self.load_loss()
        if cut_epoch is not None:
            from config import max_epochs
            loss_len = min([len(v) for v in losses.values()])
            if cut_epoch < min([max_epochs, loss_len]):
                losses = {k: v[:cut_epoch] for k, v in losses.items()}
            else:
                raise ValueError(f"cut_epoch ({cut_epoch}) is greater than loss length ({min([max_epochs, loss_len])}).")

        plt.figure(figsize=figsize)

        plt.subplot(1, 4, 1)
        plt.plot(losses['G_A2B_losses'], label='Generator A2B Loss (Train)', color='blue')
        plt.plot(losses['val_G_A2B_losses'], label='Generator A2B Loss (Val)', color='red')
        plt.title('Generator A2B Losses')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')

        plt.subplot(1, 4, 2)
        plt.plot(losses['G_B2A_losses'], label='Generator B2A Loss (Train)', color='blue')
        plt.plot(losses['val_G_B2A_losses'], label='Generator B2A Loss (Val)', color='red')
        plt.title('Generator B2A Losses')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend(loc='upper right')

        plt.subplot(1, 4, 3)
        plt.plot(losses['D_A_losses'], label='Discriminator_A Loss (Train)', color='blue')
        plt.plot(losses['val_D_A_losses'], label='Discriminator_A Loss (Val)', color='red')
        plt.title('Discriminator_A Losses')
        plt.xlabel('Epochs')
        # plt.ylabel('Loss')
        plt.legend(loc='upper right')
        
        plt.subplot(1, 4, 3)
        plt.plot(losses['D_B_losses'], label='Discriminator_B Loss (Train)', color='blue')
        plt.plot(losses['val_D_B_losses'], label='Discriminator_B Loss (Val)', color='red')
        plt.title('Discriminator_B Losses')
        plt.xlabel('Epochs')
        # plt.ylabel('Loss')
        plt.legend(loc='upper right')

        if cut_epoch is not None:
            for i in range(3):
                plt.subplot(1, 3, i+1)
                plt.xlim([0,cut_epoch-1])
                
        plt.tight_layout()
        plt.show()
