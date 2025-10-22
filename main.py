"""
Description: This script is the main entry point for the LightningCLI.
"""
import os
import sys
from itertools import chain
from pathlib import Path
from argparse import ArgumentParser
from collections import defaultdict
import yaml
import torch
import numpy as np

from lightning.pytorch.cli import LightningCLI, SaveConfigCallback
from lightning.pytorch.callbacks import BasePredictionWriter
from types import SimpleNamespace

from mri_utils import save_reconstructions
from pl_modules import PromptMrModule


def preprocess_save_dir():
    """Ensure `save_dir` exists, handling both command-line arguments and YAML configuration."""
    parser = ArgumentParser()
    parser.add_argument("--config", type=str, nargs="*",
                        help="Path(s) to YAML config file(s)")
    parser.add_argument("--trainer.logger.save_dir",
                        type=str, help="Logger save directory")
    args, _ = parser.parse_known_args(sys.argv[1:])

    save_dir = None  # Default to None

    if args.config:
        for config_path in args.config:
            if os.path.exists(config_path):
                with open(config_path, "r", encoding='utf-8') as f:
                    try:
                        config = yaml.safe_load(f)
                        if config is not None:
                            # Safely navigate to trainer.logger.save_dir
                            trainer = config.get("trainer", {})
                            logger = trainer.get("logger", {})
                            if isinstance(logger, dict) :  # Ensure logger is a dictionary
                                yaml_save_dir = logger.get(
                                    "init_args", {}).get("save_dir")
                                if yaml_save_dir:
                                    save_dir = yaml_save_dir  # Use the first valid save_dir found
                                    break
                    except yaml.YAMLError as e:
                        print(f"Error parsing YAML file {config_path}: {e}")

    for i, arg in enumerate(sys.argv):
        if arg == "--trainer.logger.save_dir":
            save_dir = sys.argv[i + 1] if i + 1 < len(sys.argv) else None
            break

    if not save_dir:
        print("Logger save_dir is None. No action taken.")
        return

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
        print(f"Pre-created logger save_dir: {save_dir}")


class CustomSaveConfigCallback(SaveConfigCallback):
    '''save the config file to the logger's run directory, merge tags from different configs'''

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.merged_tags = self._collect_tags_from_configs()

    def _collect_tags_from_configs(self):
        config_files = []
        merged_tags = set()

        for i, arg in enumerate(sys.argv):
            if arg == '--config' and i + 1 < len(sys.argv):
                config_files.append(sys.argv[i + 1])

        for config_file in config_files:
            if os.path.exists(config_file):
                try:
                    with open(config_file, 'r', encoding='utf-8') as f:
                        config_data = yaml.safe_load(f)
                        if isinstance(config_data, dict):
                            logger = config_data.get('trainer', {}).get(
                                'logger', {})
                            if logger and isinstance(logger, dict):
                                tags = logger.get('init_args', {}).get('tags', [])
                                if isinstance(tags, list):
                                    merged_tags.update(tags)
                except (yaml.YAMLError, IOError) as e:
                    print(f"Warning: Error reading {config_file}: {str(e)}")
        return merged_tags

    def setup(self, trainer, pl_module, stage):
        if hasattr(self.config, 'trainer') and hasattr(self.config.trainer, 'logger'):
            logger_config = self.config.trainer.logger
            if hasattr(logger_config, 'init_args'):
                logger_config.init_args['tags'] = list(self.merged_tags)
                if hasattr(trainer, 'logger') and trainer.logger is not None:
                    trainer.logger.experiment.tags = list(self.merged_tags)

        super().setup(trainer, pl_module, stage)

    def save_config(self, trainer, pl_module, stage) -> None:
        """Save the configuration file under the logger's run directory."""
        if stage == "predict":
            print("Skipping saving configuration in predict mode.")
            return  
        if trainer.logger is not None and hasattr(trainer.logger, "experiment"):
            project_name = trainer.logger.experiment.project_name()
            run_id = trainer.logger.experiment.id
            save_dir = trainer.logger.save_dir
            run_dir = os.path.join(save_dir, project_name, run_id)
            
            os.makedirs(run_dir, exist_ok=True)
            config_path = os.path.join(run_dir, "config.yaml")
            self.parser.save(
                self.config, config_path, skip_none=False, overwrite=self.overwrite, multifile=self.multifile
            )
            print(f"Configuration saved to {config_path}")


class CustomWriter(BasePredictionWriter):
    """
    A custom prediction writer to save reconstructions to disk.
    """

    def __init__(self, output_dir: Path, write_interval, save_masked_kspace: bool = False, masked_kspace_output_dir: str = None):
        super().__init__(write_interval)
        self.output_dir = output_dir
        self.outputs = defaultdict(list)
        self.save_masked_kspace = save_masked_kspace
        self.masked_kspace_output_dir = Path(masked_kspace_output_dir) if masked_kspace_output_dir else None
        
    def write_on_batch_end(self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx):
        """
        Collect predictions batch by batch and organize them by volume.
        Assumes `predictions` contains a dictionary with 'volume_id' and 'slice_prediction'.
        """
        pass
    
    # In main.py
    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        # --- This distributed gathering logic is from your code ---
        if torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            gathered = [None] * torch.distributed.get_world_size()
            torch.distributed.all_gather_object(gathered, predictions)
            torch.distributed.barrier()  # 添加barrier确保同步
            if not trainer.is_global_zero:
                return
            # This correctly flattens the list of lists from all GPUs
            predictions = [item for sublist in gathered for item in sublist]
        else:
            # single-GPU / no DDP: keep predictions as a flat list of dicts
            # (if it ever comes in wrapped as a list-of-lists, unwrap one level)
            if isinstance(predictions, list) and predictions and isinstance(predictions[0], list):
                predictions = predictions[0]
        
        # 只在主进程执行保存
        if not trainer.is_global_zero:
            return

        # --- Grouping logic from your code, slightly modified ---
        outputs = defaultdict(list)

        for batch_output in predictions:
            # Loop through items in the batch (handles batch_size > 1)
            for i in range(len(batch_output["fname"])):
                fname = batch_output["fname"][i]
                # Ensure fname is a string, not a tuple
                if isinstance(fname, tuple):
                    fname = fname[0]
                
                # Store the entire dictionary for this slice
                outputs[fname].append({
                    "slice_num": batch_output["slice_num"][i],
                    "time_frame": batch_output["time_frame"][i],
                    "num_slc": batch_output["num_slc"][i],
                    "output": batch_output["output"][i],
                    "has_fake_time_dim": batch_output["has_fake_time_dim"][i],
                    "mask": batch_output.get("mask", [None])[i] if "mask" in batch_output else None,
                    "mask_type": batch_output.get("mask_type", [None])[i] if "mask_type" in batch_output else None,
                    "num_low_frequencies": batch_output.get("num_low_frequencies", [None])[i] if "num_low_frequencies" in batch_output else None,
                    "masked_kspace": batch_output.get("masked_kspace", [None])[i] if "masked_kspace" in batch_output else None,
                    "original_kspace": batch_output.get("kspace", [None])[i] if "kspace" in batch_output else None,
                    "attrs": batch_output.get("attrs", [None])[i] if "attrs" in batch_output else None
                })
                
        # Directory for saving the final volumes
        save_dir = self.output_dir / "TaskS2/MultiCoil"
        save_dir.mkdir(parents=True, exist_ok=True)

        # --- FIX #1: This entire block replaces your old "Sort and Stack" logic ---
        # Process each file independently to prevent metadata mix-ups
        for fname, slice_data_list in outputs.items():
            if not slice_data_list:
                continue

            try:
                # a. Sort all slices for THIS file by time, then by slice number
                sorted_slices = sorted(
                    slice_data_list,
                    key=lambda s: (s['time_frame'].item(), s['slice_num'].item())
                )

                # b. Stack all sorted slices into one tensor
                stacked_volume = torch.stack([s['output'].squeeze() for s in sorted_slices])
                
                # b1. Stack masks if available
                stacked_masks = None
                if sorted_slices[0]['mask'] is not None:
                    stacked_masks = torch.stack([s['mask'].squeeze() for s in sorted_slices])
                
                # b2. Stack masked k-space if available
                stacked_masked_kspace = None
                if sorted_slices[0]['masked_kspace'] is not None:
                    stacked_masked_kspace = torch.stack([s['masked_kspace'].squeeze() for s in sorted_slices])

                # c. Get the correct number of slices for THIS file from its own data
                num_slices_per_time = sorted_slices[0]['num_slc'].item()
                num_time_frames = len(sorted_slices) // num_slices_per_time
                has_fake_time_dim = sorted_slices[0]['has_fake_time_dim']
                
                # c1. Collect mask metadata
                mask_types = [s['mask_type'] for s in sorted_slices if s['mask_type'] is not None]
                num_low_freqs = [s['num_low_frequencies'] for s in sorted_slices if s['num_low_frequencies'] is not None]
                
                # d. Final check to ensure data is consistent
                if len(sorted_slices) % num_slices_per_time != 0:
                    print(f"Warning: Inconsistent data for {fname}. Skipping.")
                    continue
                
                h, w = stacked_volume.shape[-2], stacked_volume.shape[-1]
                
                # e. Reshape into the final 4D volume using the CORRECT dimensions
                final_4d_volume = stacked_volume.view(num_time_frames, num_slices_per_time, h, w)

                # f. Check if we need to remove fake time dimension 
                if has_fake_time_dim and num_time_frames == 2:
                    # Original data was 4D, fake time dimension was added, remove it
                    print(f"Removing fake time dimension for {fname}: {final_4d_volume.shape} -> 3D")
                    # Take the first time frame and remove the time dimension
                    final_volume = final_4d_volume[0]  # Shape: (num_slices_per_time, h, w)
                    print(f"Saving 3D volume for {fname} with shape {final_volume.shape}")
                    save_reconstructions(final_volume, fname, save_dir)
                else:
                    print(f"Saving 4D volume for {fname} with shape {final_4d_volume.shape}")
                    # The save function is now much simpler
                    save_reconstructions(final_4d_volume, fname, save_dir)
                
                # Save masked k-space and masks if enabled
                if self.save_masked_kspace and stacked_masked_kspace is not None:
                    self._save_masked_kspace_data(fname, sorted_slices, stacked_masked_kspace, stacked_masks, 
                                                num_time_frames, num_slices_per_time, has_fake_time_dim)

            except Exception as e:
                print(f"CRITICAL ERROR while processing/saving {fname}. Error: {e}")
                
        print(f"Done! Reconstructions saved to {save_dir}")
    
    def _save_masked_kspace_data(self, fname, sorted_slices, stacked_masked_kspace, stacked_masks, 
                                num_time_frames, num_slices_per_time, has_fake_time_dim):
        """Save masked k-space and masks in the specified format"""
        import scipy.io as sio
        import os
        
        # Extract path components from fname
        # Expected format: /path/to/Center010/UIH_30T_umr790/P060/lge_lax_4ch.mat
        fname_parts = fname.split('/')
        
        # Find the base filename and extract components
        base_fname = os.path.basename(fname)
        name_without_ext = base_fname.replace('.mat', '')
        
        # Extract center, scanner, patient info from path
        center_idx = -1
        scanner_idx = -1
        patient_idx = -1
        
        for i, part in enumerate(fname_parts):
            if part.startswith('Center'):
                center_idx = i
            elif 'UIH' in part or 'Siemens' in part or 'GE' in part:
                scanner_idx = i
            elif part.startswith('P'):
                patient_idx = i
        
        if center_idx == -1 or scanner_idx == -1 or patient_idx == -1:
            print(f"Warning: Could not parse path components from {fname}")
            return
        
        center = fname_parts[center_idx]
        scanner = fname_parts[scanner_idx]
        patient = fname_parts[patient_idx]
        
        # Get mask type and acceleration info from the first slice
        mask_type = sorted_slices[0]['mask_type']
        num_low_freq = sorted_slices[0]['num_low_frequencies']
        
        # Create mask type suffix
        if mask_type == 'kt_uniform':
            mask_suffix = f"ktUniform{num_low_freq}"
        elif mask_type == 'kt_random':
            mask_suffix = f"ktRandom{num_low_freq}"
        elif mask_type == 'kt_radial':
            mask_suffix = f"ktRadial{num_low_freq}"
        elif mask_type == 'uniform':
            mask_suffix = f"Uniform{num_low_freq}"
        else:
            mask_suffix = f"{mask_type}{num_low_freq}"
        
        # Create directory structure
        if self.masked_kspace_output_dir is not None:
            base_dir = self.masked_kspace_output_dir / "train_val/TaskS2/MultiCoil"
        else:
            # Default fallback path
            base_dir = Path("/cmrchallenge/data/CMR2025/Validation/Task4/TaskS2/MultiCoil")
        
        # Determine sequence type from filename
        if 'lge' in name_without_ext.lower():
            sequence = 'LGE'
        elif 'cine' in name_without_ext.lower():
            sequence = 'CINE'
        elif 't1' in name_without_ext.lower():
            sequence = 'T1'
        elif 't2' in name_without_ext.lower():
            sequence = 'T2'
        else:
            sequence = 'OTHER'
        
        # Create paths
        mask_dir = base_dir / sequence / "ValidationSet" / "Mask_TaskS2" / center / scanner / patient
        kspace_dir = base_dir / sequence / "ValidationSet" / "UnderSample_TaskS2" / center / scanner / patient
        
        mask_dir.mkdir(parents=True, exist_ok=True)
        kspace_dir.mkdir(parents=True, exist_ok=True)
        
        # Reshape data to match validation inference format
        # masked k-space: [Ny, Nx, Ncoil, Nz, Nt]
        # mask: [Ny, Nx, Nt] or [Ny, Nx] for 2D
        
        # Get dimensions
        if len(stacked_masked_kspace.shape) == 5:  # [batch, coil, height, width, 1]
            _, n_coils, height, width, _ = stacked_masked_kspace.shape
            stacked_masked_kspace = stacked_masked_kspace.squeeze(-1)  # Remove last dimension
        else:
            _, n_coils, height, width = stacked_masked_kspace.shape
            
        if len(stacked_masks.shape) == 4:  # [batch, height, width, 1]
            _, mask_height, mask_width, _ = stacked_masks.shape
            stacked_masks = stacked_masks.squeeze(-1)  # Remove last dimension
        else:
            _, mask_height, mask_width = stacked_masks.shape
        
        # Reshape to [num_time_frames, num_slices_per_time, ...]
        final_masked_kspace = stacked_masked_kspace.view(num_time_frames, num_slices_per_time, n_coils, height, width)
        final_masks = stacked_masks.view(num_time_frames, num_slices_per_time, mask_height, mask_width)
        
        # Remove fake time dimension if needed
        if has_fake_time_dim and num_time_frames == 2:
            final_masked_kspace = final_masked_kspace[0]  # [num_slices_per_time, n_coils, height, width]
            final_masks = final_masks[0]  # [num_slices_per_time, mask_height, mask_width]
        
        # Convert to the expected format:
        # masked k-space: [Ny, Nx, Ncoil, Nz, Nt] -> [height, width, n_coils, num_slices_per_time, num_time_frames]
        # mask: [Ny, Nx, Nt] -> [mask_height, mask_width, num_time_frames] or [Ny, Nx] -> [mask_height, mask_width]
        
        if has_fake_time_dim and num_time_frames == 2:
            # 3D case: [num_slices_per_time, n_coils, height, width] -> [height, width, n_coils, num_slices_per_time]
            final_masked_kspace = final_masked_kspace.permute(2, 3, 1, 0)  # [height, width, n_coils, num_slices_per_time]
            # 3D mask: [num_slices_per_time, mask_height, mask_width] -> [mask_height, mask_width] (2D mask)
            final_masks = final_masks[0]  # Take first slice as representative mask
        else:
            # 4D case: [num_time_frames, num_slices_per_time, n_coils, height, width] -> [height, width, n_coils, num_slices_per_time, num_time_frames]
            final_masked_kspace = final_masked_kspace.permute(3, 4, 2, 1, 0)  # [height, width, n_coils, num_slices_per_time, num_time_frames]
            # 4D mask: [num_time_frames, num_slices_per_time, mask_height, mask_width] -> [mask_height, mask_width, num_time_frames]
            final_masks = final_masks.permute(2, 3, 0)  # [mask_height, mask_width, num_time_frames]
        
        # Create filenames
        mask_fname = f"{name_without_ext}_mask_{mask_suffix}.mat"
        kspace_fname = f"{name_without_ext}_kus_{mask_suffix}.mat"
        
        # Save mask file
        mask_path = mask_dir / mask_fname
        sio.savemat(mask_path, {
            'mask': final_masks.numpy(),
            'mask_type': mask_type,
            'num_low_frequencies': num_low_freq
        })
        print(f"Saved mask to {mask_path} with shape {final_masks.shape}")
        
        # Save masked k-space file
        kspace_path = kspace_dir / kspace_fname
        sio.savemat(kspace_path, {
            'kus': final_masked_kspace.numpy(),  # Use 'kus' as the array name to match validation format
            'mask_type': mask_type,
            'num_low_frequencies': num_low_freq,
            'acceleration': 4  # Default acceleration, could be extracted from mask
        })
        print(f"Saved masked k-space to {kspace_path} with shape {final_masked_kspace.shape}")


class CustomLightningCLI(LightningCLI):

    def instantiate_classes(self):
        super().instantiate_classes()
        if self.config_init.subcommand == 'predict':
            ckpt_path_to_load = self.config_init.predict.ckpt_path
            print(f"\n✅ LOADING CHECKPOINT FROM: {ckpt_path_to_load}\n")
            # ✂️ DON’T call load_from_checkpoint here!
            # self.model = PromptMrModule.load_from_checkpoint(self.config_init.predict.ckpt_path)


def run_cli():
    preprocess_save_dir()

    cli = CustomLightningCLI(
        save_config_callback=CustomSaveConfigCallback,
        save_config_kwargs={"overwrite": True},
        run=True  # Let Lightning handle predict()
    )

    
if __name__ == "__main__":
    run_cli()

