# dataset/data_split.py
import sys
import pickle
from sklearn.model_selection import train_test_split
from config import root_dir
from config import train_ratio, test_ratio, val_ratio, random_state

def data_split(data_dir_list, save_dir=f'{root_dir}/dataset', savename_suffix=None):
    # Split Data into Train/Val/Test
    train_data, test_data = train_test_split(data_dir_list, test_size=test_ratio, random_state=random_state)
    train_data, val_data = train_test_split(train_data, test_size=val_ratio/(1-test_ratio), random_state=random_state)

    # Verify the Split
    print(f'Train set: {len(train_data)} samples')
    print(f'Validation set: {len(val_data)} samples')
    print(f'Test set: {len(test_data)} samples')

    suffix = savename_suffix if savename_suffix is not None else ''

    # Save Split Data
    with open(f'{save_dir}/train_data{suffix}.pkl', 'wb') as f:
        pickle.dump(train_data, f)
    with open(f'{save_dir}/val_data{suffix}.pkl', 'wb') as f:
        pickle.dump(val_data, f)
    with open(f'{save_dir}/test_data{suffix}.pkl', 'wb') as f:
        pickle.dump(test_data, f)
