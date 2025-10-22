import os
import json
import shutil
import argparse
from os.path import join
from pathlib import Path

import h5py
import numpy as np
import torch
from tqdm import tqdm
from huggingface_hub import snapshot_download

from data import to_tensor
from mri_utils import ifft2c, fft2c, complex_abs, rss

def load_masks(mask_folder, acceleration):

    masks_path = Path(mask_folder)
    paths = [
        f"R{acceleration}_218x170.npy",
        f"R{acceleration}_218x174.npy",
        f"R{acceleration}_218x180.npy",
    ]

    output = {}
    for path in paths:
        shape = path.split("_")[-1][:-4].split("x")[-1]
        mask_array = np.load(masks_path / path)
        output[shape] = mask_array.astype(np.float32), mask_array.shape[0]
    return output

def get_rss_from_kspace(kspace_file, return_kspace=False, cut_out_first_last_50_slices=True, sr=0.85,device='cuda'):
    '''
    kspace_file: str, path to kspace file
    return_kspace: bool, whether to return kspace
    cut_out_first_last_50_slices: bool, whether to cut out first and last 50 slices
    sr: float, partial fourier percentage in the slice-encoded direction
    '''

    with h5py.File(kspace_file,'r') as hf:
        # cut out first and last 50 slices
        if cut_out_first_last_50_slices:
            kspace = hf['kspace'][50:-50]
        else:
            kspace = hf['kspace'][()]
    # 1. correct the kdata-img mapping
    kspace = kspace[...,0::2]+1j*kspace[...,1::2]
    kspace = kspace.transpose(3,0,1,2) # [12, 256, 218, 170]
    kspace_th = to_tensor(kspace) # [12, 256, 218, 170, 2]
    img = torch.view_as_real(
            torch.fft.ifft2(  # type: ignore
                torch.fft.ifftshift(torch.view_as_complex(kspace_th.to(device)), dim=(-2,-1)), dim=(-2, -1) , norm='ortho'
            ))

    # 2. Explicit zero-filling after 85% in the slice-encoded direction
    kspace = fft2c(img)
    Nz = kspace.shape[3]
    Nz_sampled = int(np.ceil(Nz*sr))
    kspace[:,:,:,Nz_sampled:,:] = 0

    # 3. get img
    img = ifft2c(kspace)
    img_rss= rss(complex_abs(img),dim=0)

    kspace = torch.view_as_complex(kspace)

    if return_kspace:
        return img_rss.cpu().numpy(), kspace.cpu().numpy().transpose(1,0,2,3)
    else:
        return img_rss.cpu().numpy()
    
def prepare_fullysample_file(raw_folder, save_folder, ff, device):
    ff = ff.split('/')[-1] 
    raw_path = join(raw_folder, ff)
    save_path = join(save_folder, ff)

    img_rss, kdata = get_rss_from_kspace(raw_path, return_kspace=True, cut_out_first_last_50_slices=False, device=device) # (slice, height, width), (slice, coil, height, width)

    # Open the HDF5 file in write mode
    with h5py.File(save_path, 'w') as file:
        # Create a dataset
        save_kdata = kdata
        file.create_dataset('kspace', data=save_kdata)

        file.create_dataset('reconstruction_rss', data=img_rss)
        file.attrs['max'] = img_rss.max()
        file.attrs['norm'] = np.linalg.norm(img_rss)

        # Add attributes to the dataset
        file.attrs['acquisition'] = '3D_GRE_T1'
        file.attrs['patient_id'] = save_path.split('/')[-1].split('.h5')[0]
        file.attrs['shape'] = kdata.shape
        file.attrs['padding_left'] = 0
        file.attrs['padding_right'] = save_kdata.shape[3]
        file.attrs['encoding_size'] = (save_kdata.shape[2],save_kdata.shape[3],1)
        file.attrs['recon_size'] = (save_kdata.shape[2],save_kdata.shape[3],1)
        print(img_rss.shape, save_kdata.shape)

def prepare_test_for_file(kspace_full, test_save_path, masks_bank, acc, mask_idx_list):
    num_cols = kspace_full.shape[-1]
    with h5py.File(test_save_path, 'w') as ht:
        mask_, _ = masks_bank[acc][str(num_cols)]
        mask_ = mask_[mask_idx_list[ii]]
        masked_kspace = kspace_full*mask_[None,None]+0.0
        ht.create_dataset('kspace', data=masked_kspace)
        ht.create_dataset('mask', data=mask_)
        ht.attrs['acquisition'] = '3D_GRE_T1'
        ht.attrs['patient_id'] = test_save_path.split('/')[-1].split('.h5')[0]
        ht.attrs['acceleration'] = acc
        ht.attrs['num_low_frequency'] = 18
        ht.attrs['padding_left'] = 0
        ht.attrs['padding_right'] = masked_kspace.shape[3]
        ht.attrs['recon_size'] = (masked_kspace.shape[2],masked_kspace.shape[3],1)
               
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Prepare H5 dataset for calgary-campinas dataset')
    parser.add_argument('--output_folder', type=str,
                        default='/gpfs/data/axellab/bingyx01/dataset/cc-brain',
                        help='path to save H5 dataset')
    parser.add_argument('--input_folder', type=str,
                        default='/gpfs/data/axellab/bingyx01/dataset/calgary-campinas_version-1.0/CC359/Raw-data/Multi-channel/12-channel',
                        help='orginal raw data folder')
    parser.add_argument('--split_json', type=str,
                        default='configs/data_split/cc-brain.json',
                        help='path to split json file')
    
    args = parser.parse_args()

    # check if cuda is available
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # 1. define folders
    raw_train_folder = join(args.input_folder, 'Train')
    raw_val_folder = join(args.input_folder, 'Val')
        
    # save folders
    save_train_folder = join(args.output_folder, 'train')
    save_val_folder = join(args.output_folder, 'val')
    save_test_full_folder = join(args.output_folder, 'test_full')
    save_test_acc5_folder = join(args.output_folder, 'test_acc05')
    save_test_acc10_folder = join(args.output_folder, 'test_acc10') 
    if not os.path.exists(save_train_folder):
        os.makedirs(save_train_folder)
    if not os.path.exists(save_val_folder):
        os.makedirs(save_val_folder)
    if not os.path.exists(save_test_full_folder):
        os.makedirs(save_test_full_folder)
    if not os.path.exists(save_test_acc5_folder):
        os.makedirs(save_test_acc5_folder)
    if not os.path.exists(save_test_acc10_folder):
        os.makedirs(save_test_acc10_folder)

    # 2. download mask files
    download_folder = join(args.output_folder, 'mask_files_required_for_training/poisson_sampling')
    mask_folder = join(args.output_folder, 'poisson_sampling')
    if not os.path.exists(mask_folder):
        _ = snapshot_download(
            repo_id='hellopipu/PromptMR',
            local_dir = args.output_folder,
            allow_patterns = ['mask_files_required_for_training/poisson_sampling/*']
        )
        shutil.move(download_folder, mask_folder)
        shutil.rmtree(join(args.output_folder, 'mask_files_required_for_training'))
    
    masks_bank = {5: load_masks(mask_folder,5), 10: load_masks(mask_folder, 10)}
    
    # 3. load split json
    with open(args.split_json, 'r') as f:
        loaded_data = json.load(f)
    # Retrieve the split lists
    train_file_list = loaded_data['train']
    val_file_list = loaded_data['val']
    test_file_list = loaded_data['test_acc05']
    test_mask_index_list = loaded_data['test_mask_indices']
    
    # 4. prepare fully sampled data
    # 4.1 prepare train data
    for ff in tqdm(train_file_list):
        prepare_fullysample_file(raw_train_folder, save_train_folder, ff, device)
    # 4.2 prepare val data
    for ff in tqdm(val_file_list):
        prepare_fullysample_file(raw_val_folder, save_val_folder, ff, device)
    # 4.3 prepare test data
    for ff in tqdm(test_file_list):
        prepare_fullysample_file(raw_val_folder, save_test_full_folder, ff, device)

    # 5. prepare undersampled test data
    for ii,ff in tqdm(enumerate(test_file_list)):
        ff = ff.split('/')[-1]
        ff_test_path_5 = join(save_test_acc5_folder, ff)
        ff_test_path_10 = join(save_test_acc10_folder, ff)

        ff_full_test_path = join(save_test_full_folder, ff)
        with h5py.File(ff_full_test_path, 'r') as hf:
            kspace_full = hf['kspace'][()]

        acc05, acc10 = 5, 10
        # prepare undersampled test data
        prepare_test_for_file(kspace_full, ff_test_path_5, masks_bank, acc05, test_mask_index_list)
        prepare_test_for_file(kspace_full, ff_test_path_10, masks_bank, acc10, test_mask_index_list)


