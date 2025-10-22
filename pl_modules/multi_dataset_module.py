from pathlib import Path
from typing import Callable, Optional, Union, Type, List
import importlib
import lightning as L
import torch
import torch.utils
from data import CombinedSliceDataset, VolumeSampler, FastmriSliceDataset
from data import InferVolumeBatchSampler, InferVolumeDistributedSampler
from data import RawDataSample
import logging


#########################################################################################################
# Common functions
#########################################################################################################

def worker_init_fn(worker_id):
    """Handle random seeding for all mask_func."""
    worker_info = torch.utils.data.get_worker_info()
    data: Union[
        torch.utils.data.Dataset, CombinedSliceDataset
    ] = worker_info.dataset  # pylint: disable=no-member

    # Check if we are using DDP
    is_ddp = False
    if torch.distributed.is_available():
        if torch.distributed.is_initialized():
            is_ddp = True

    # for NumPy random seed we need it to be in this range
    base_seed = worker_info.seed  # pylint: disable=no-member

    if isinstance(data, CombinedSliceDataset):
        for i, dataset in enumerate(data.datasets):
            if dataset.transform.mask_func is not None:
                if (
                    is_ddp
                ):  # DDP training: unique seed is determined by worker, device, dataset
                    seed_i = (
                        base_seed
                        - worker_info.id
                        + torch.distributed.get_rank()
                        * (worker_info.num_workers * len(data.datasets))
                        + worker_info.id * len(data.datasets)
                        + i
                    )
                else:
                    seed_i = (
                        base_seed
                        - worker_info.id
                        + worker_info.id * len(data.datasets)
                        + i
                    )
                dataset.transform.mask_func.rng.seed(seed_i % (2**32 - 1))
    elif data.transform.mask_func is not None:
        if is_ddp:  # DDP training: unique seed is determined by worker and device
            seed = base_seed + torch.distributed.get_rank() * worker_info.num_workers
        else:
            seed = base_seed
        data.transform.mask_func.rng.seed(seed % (2**32 - 1))

def _check_both_not_none(val1, val2):
    if (val1 is not None) and (val2 is not None):
        return True
    return False

def resolve_class(class_path: str):
    """Dynamically resolve a class from its string path."""
    module_name, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


#########################################################################################################
# Multi-Dataset Balance Sampler
#########################################################################################################

class MultiDatasetBalanceSampler:
    """
    Balance sampler that can handle multiple datasets with different naming conventions.
    Uses dataset-specific balancing strategies for 2024 and 2025 datasets.
    """
    
    def __init__(self, ratio_dict_2024: dict = None, ratio_dict_2025: dict = None, unified_ratio_dict: dict = None):
        """
        Initialize with dataset-specific ratio dictionaries.
        
        Args:
            ratio_dict_2024: Balancing ratios for 2024 dataset
            ratio_dict_2025: Balancing ratios for 2025 dataset  
            unified_ratio_dict: Fallback unified ratios (for backward compatibility)
        """
        # Dataset-specific ratios
        self.ratio_dict_2024 = ratio_dict_2024 or {
            'T1map': 2, 
            'T2map': 6, 
            'cine_lax': 2, 
            'cine_sax': 1, 
            'cine_lvot': 6, 
            'aorta_sag': 1, 
            'aorta_tra': 1,
            'tagging': 1
        }
        
        self.ratio_dict_2025 = ratio_dict_2025 or {
            'cine_rvot': 8,
            'cine_sax': 1,
            'lge_lax_4ch': 8,
            'flow2d': 3,
            'cine_lax': 8,
            'T1w': 4,
            'lge_sax': 2,
            'T2map': 4,
            'perfusion': 8,
            'T1rho': 8,
            'T1map': 3,
            'cine_lax_3ch': 8,
            'lge_lax_2ch': 8,
            'cine_lax_2ch': 8,
            'T1mappost': 8,
            'T2w': 2,
            'cine_lax_4ch': 8,
            'lge_lax_3ch': 8,
            'blackblood': 8,
            'cine_lvot': 8,
            'cine_ot': 8,
            'lge_lax': 8,
            'cine_lax_r2ch': 8,
            'T2smap': 8,
        }
        
        # Fallback unified ratios (for backward compatibility)
        self.unified_ratio_dict = unified_ratio_dict
        
        # Mapping from 2024 dataset names to 2025 dataset names
        self.name_mapping_2024_to_2025 = {
            'cine_lvot': 'cine_lvot',
            'cine_sax': 'cine_sax', 
            'cine_lax': 'cine_lax',
            'T1map': 'T1map',
            'T2map': 'T2map',
            'aorta_sag': 'aorta_sag',
            'tagging': 'tagging',
            'T1w': 'T1w',
            'T2w': 'T2w',
            'lge': 'lge',
            'perfusion': 'perfusion',
            'T1rho': 'T1rho',
            'blackblood': 'blackblood',
            'flow2d': 'flow2d',
            'T1mappost': 'T1mappost',
            'T2smap': 'T2smap',
        }
        
        # Reverse mapping for 2025 to 2024
        self.name_mapping_2025_to_2024 = {v: k for k, v in self.name_mapping_2024_to_2025.items()}
        
    def _detect_dataset_type(self, filename: str) -> str:
        """Detect if the file is from 2024 or 2025 dataset."""
        filename = str(filename)
        if 'Center' in filename:
            return '2025'
        else:
            return '2024'
        
    def _extract_sequence_type(self, filename: str) -> str:
        """Extract sequence type from filename, handling both 2024 and 2025 naming conventions."""
        filename = str(filename)
        
        # Check if it's 2025 dataset (has Center prefix)
        if 'Center' in filename:
            # 2025 format: Center001_UIH_30T_umr780_P001_cine_lax_3ch.h5
            parts = filename.split('_')
            # Find the sequence type part (usually after P001)
            for i, part in enumerate(parts):
                if part.startswith('P') and part[1:].isdigit():
                    if i + 1 < len(parts):
                        sequence_parts = parts[i+1:]
                        # Remove .h5 extension
                        sequence_parts[-1] = sequence_parts[-1].replace('.h5', '')
                        sequence_type = '_'.join(sequence_parts)
                        return sequence_type
        else:
            # 2024 format: P001_cine_lvot.h5
            parts = filename.split('_')
            if len(parts) >= 2:
                sequence_parts = parts[1:]
                # Remove .h5 extension
                sequence_parts[-1] = sequence_parts[-1].replace('.h5', '')
                sequence_type = '_'.join(sequence_parts)
                return sequence_type
        
        return 'unknown'
    
    def _normalize_sequence_type(self, sequence_type: str, dataset_type: str) -> str:
        """Normalize sequence type to match the appropriate ratio_dict keys."""
        # Get the appropriate ratio dict for this dataset
        if dataset_type == '2024':
            ratio_dict = self.ratio_dict_2024
        elif dataset_type == '2025':
            ratio_dict = self.ratio_dict_2025
        else:
            # Fallback to unified dict if available
            ratio_dict = self.unified_ratio_dict or self.ratio_dict_2025
        
        # Try direct match first
        if sequence_type in ratio_dict:
            return sequence_type
            
        # Try mapping from 2024 to 2025
        if sequence_type in self.name_mapping_2024_to_2025:
            normalized = self.name_mapping_2024_to_2025[sequence_type]
            if normalized in ratio_dict:
                return normalized
                
        # Try mapping from 2025 to 2024
        if sequence_type in self.name_mapping_2025_to_2024:
            normalized = self.name_mapping_2025_to_2024[sequence_type]
            if normalized in ratio_dict:
                return normalized
        
        # Try partial matching
        for key in ratio_dict.keys():
            if key in sequence_type or sequence_type in key:
                return key
                
        return 'unknown'
    
    def _get_ratio_for_sample(self, sequence_type: str, dataset_type: str) -> int:
        """Get the appropriate ratio for a sample based on its sequence type and dataset."""
        normalized_type = self._normalize_sequence_type(sequence_type, dataset_type)
        
        # Get the appropriate ratio dict
        if dataset_type == '2024':
            ratio_dict = self.ratio_dict_2024
        elif dataset_type == '2025':
            ratio_dict = self.ratio_dict_2025
        else:
            # Fallback to unified dict if available
            ratio_dict = self.unified_ratio_dict or self.ratio_dict_2025
        
        if normalized_type in ratio_dict:
            return ratio_dict[normalized_type]
        
        return 1  # Default ratio for unknown types
    
    def __call__(self, raw_samples: List[RawDataSample]) -> List[RawDataSample]:
        """Balance the dataset based on sequence types with dataset-specific handling."""
        # Create dict with empty lists for each sequence type and dataset
        dict_list_2024 = {key: [] for key in self.ratio_dict_2024.keys()}
        dict_list_2025 = {key: [] for key in self.ratio_dict_2025.keys()}
        dict_list_2024['unknown'] = []
        dict_list_2025['unknown'] = []
        
        # Categorize samples by sequence type and dataset
        for raw_sample in raw_samples:
            sequence_type = self._extract_sequence_type(raw_sample.fname)
            dataset_type = self._detect_dataset_type(raw_sample.fname)
            
            if dataset_type == '2024':
                normalized_type = self._normalize_sequence_type(sequence_type, '2024')
                if normalized_type in dict_list_2024:
                    dict_list_2024[normalized_type].append(raw_sample)
                else:
                    dict_list_2024['unknown'].append(raw_sample)
            else:  # 2025
                normalized_type = self._normalize_sequence_type(sequence_type, '2025')
                if normalized_type in dict_list_2025:
                    dict_list_2025[normalized_type].append(raw_sample)
                else:
                    dict_list_2025['unknown'].append(raw_sample)
        
        # Combine samples with dataset-specific ratios
        final_list = []
        
        # Process 2024 samples
        for key in self.ratio_dict_2024.keys():
            if key in dict_list_2024 and dict_list_2024[key]:
                ratio = self._get_ratio_for_sample(key, '2024')
                final_list += dict_list_2024[key] * ratio
                logging.info(f"2024 - Sequence type '{key}': {len(dict_list_2024[key])} samples, ratio {ratio}")
        
        # Process 2025 samples
        for key in self.ratio_dict_2025.keys():
            if key in dict_list_2025 and dict_list_2025[key]:
                ratio = self._get_ratio_for_sample(key, '2025')
                final_list += dict_list_2025[key] * ratio
                logging.info(f"2025 - Sequence type '{key}': {len(dict_list_2025[key])} samples, ratio {ratio}")
        
        # Add unknown samples with ratio 1
        if dict_list_2024['unknown']:
            final_list += dict_list_2024['unknown']
            logging.info(f"2024 - Unknown sequence types: {len(dict_list_2024['unknown'])} samples")
        
        if dict_list_2025['unknown']:
            final_list += dict_list_2025['unknown']
            logging.info(f"2025 - Unknown sequence types: {len(dict_list_2025['unknown'])} samples")
        
        logging.info(f"Total balanced samples: {len(final_list)}")
        return final_list


#########################################################################################################
# Multi-Dataset DataModule
#########################################################################################################

class MultiDatasetDataModule(L.LightningDataModule):
    """
    Data module for training on multiple datasets (2024 and 2025).
    Supports different naming conventions and unified data balancing.
    """

    def __init__(
        self,
        slice_dataset: str,
        data_paths: List[Path],  # List of data paths for different datasets
        challenge: str,
        train_transform: Callable,
        val_transform: Callable,
        combine_train_val: bool = False,
        sample_rate: Optional[float] = None,
        val_sample_rate: Optional[float] = None,
        volume_sample_rate: Optional[float] = None,
        val_volume_sample_rate: Optional[float] = None,
        train_filter: Optional[Callable] = None,
        val_filter: Optional[Callable] = None,
        use_dataset_cache_file: bool = True,
        batch_size: int = 1,
        num_workers: int = 4,
        distributed_sampler: bool = False,
        num_adj_slices: int = 5,
        data_balancer: Optional[Callable] = None,
        use_pre_whiten: bool = False,
    ):
        super().__init__()

        if _check_both_not_none(sample_rate, volume_sample_rate):
            raise ValueError("Can set sample_rate or volume_sample_rate, but not both.")
        if _check_both_not_none(val_sample_rate, val_volume_sample_rate):
            raise ValueError("Can set val_sample_rate or val_volume_sample_rate, but not both.")

        self.slice_dataset = resolve_class(slice_dataset)
        self.data_paths = data_paths
        self.challenge = challenge
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.combine_train_val = combine_train_val
        self.sample_rate = sample_rate
        self.val_sample_rate = val_sample_rate
        self.volume_sample_rate = volume_sample_rate
        self.val_volume_sample_rate = val_volume_sample_rate
        self.train_filter = train_filter
        self.val_filter = val_filter
        self.use_dataset_cache_file = use_dataset_cache_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.distributed_sampler = distributed_sampler
        self.num_adj_slices = num_adj_slices
        self.data_balancer = data_balancer
        self.use_pre_whiten = use_pre_whiten

    def _create_data_loader(
        self,
        slice_dataset: Type,
        data_transform: Callable,
        data_partition: str,
        sample_rate: Optional[float] = None,
        volume_sample_rate: Optional[float] = None,
    ) -> torch.utils.data.DataLoader:
        raw_sample_filter: Optional[Callable]
        if data_partition == "train":
            is_train = True
            sample_rate = self.sample_rate if sample_rate is None else sample_rate
            volume_sample_rate = (
                self.volume_sample_rate
                if volume_sample_rate is None
                else volume_sample_rate
            )
            raw_sample_filter = self.train_filter
        else:
            is_train = False
            if data_partition == "val":
                sample_rate = (
                    self.val_sample_rate if sample_rate is None else sample_rate
                )
                volume_sample_rate = (
                    self.val_volume_sample_rate
                    if volume_sample_rate is None
                    else volume_sample_rate
                )
                raw_sample_filter = self.val_filter

        # Create data paths for all datasets
        data_paths = []
        for data_path in self.data_paths:
            prefix = f"{self.challenge}_" if self.slice_dataset is FastmriSliceDataset else ""
            data_paths.append(data_path / f"{prefix}{data_partition}")

        # Create datasets for each data path
        datasets = []
        for data_path in data_paths:
            if data_path.exists():
                dataset = slice_dataset(
                    root=data_path,
                    transform=data_transform,
                    sample_rate=sample_rate,
                    volume_sample_rate=volume_sample_rate,
                    challenge=self.challenge,
                    use_dataset_cache=self.use_dataset_cache_file,
                    raw_sample_filter=raw_sample_filter,
                    data_balancer=self.data_balancer if is_train else None,
                    num_adj_slices=self.num_adj_slices,
                    use_pre_whiten=self.use_pre_whiten,
                )
                datasets.append(dataset)
                logging.info(f"Created dataset from {data_path} with {len(dataset)} samples")
            else:
                logging.warning(f"Data path does not exist: {data_path}")

        if not datasets:
            raise ValueError("No valid datasets found!")

        # Combine datasets
        if len(datasets) == 1:
            dataset = datasets[0]
        else:
            dataset = CombinedSliceDataset(
                slice_dataset=slice_dataset,
                roots=data_paths,
                transforms=[data_transform] * len(data_paths),
                challenges=[self.challenge] * len(data_paths),
                sample_rates=[sample_rate] * len(data_paths) if sample_rate is not None else None,
                volume_sample_rates=[volume_sample_rate] * len(data_paths) if volume_sample_rate is not None else None,
                use_dataset_cache=self.use_dataset_cache_file,
                raw_sample_filter=raw_sample_filter,
                data_balancer=self.data_balancer if is_train else None,
                num_adj_slices=self.num_adj_slices,
            )

        # Ensure that entire volumes go to the same GPU in the ddp setting
        sampler = None

        if self.distributed_sampler:
            if is_train:
                sampler = torch.utils.data.DistributedSampler(dataset)
            else:
                sampler = VolumeSampler(dataset, shuffle=False)

        dataloader = torch.utils.data.DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            worker_init_fn=worker_init_fn,
            sampler=sampler,
            shuffle=is_train if sampler is None else False,
        )

        return dataloader

    def prepare_data(self):
        """Prepare data for all datasets."""
        if self.use_dataset_cache_file:
            prefix = f"{self.challenge}_" if self.slice_dataset is FastmriSliceDataset else ""

            for data_path in self.data_paths:
                data_paths = [
                    data_path / f"{prefix}train",
                    data_path / f"{prefix}val",
                ]

                data_transforms = [
                    self.train_transform,
                    self.val_transform,
                ]
                
                raw_sample_filters = [
                    self.train_filter,
                    self.val_filter,
                ]
                
                data_balancers = [
                    self.data_balancer,
                    None,
                ]
                
                for i, (data_path, data_transform, raw_sample_filter, data_balancer) in enumerate(
                    zip(data_paths, data_transforms, raw_sample_filters, data_balancers)
                ):
                    if data_path.exists():
                        # NOTE: Fixed so that val and test use correct sample rates
                        sample_rate = self.sample_rate  # if i == 0 else 1.0
                        volume_sample_rate = self.volume_sample_rate  # if i == 0 else None
                        _ = self.slice_dataset(
                            root=data_path,
                            transform=data_transform,
                            sample_rate=sample_rate,
                            volume_sample_rate=volume_sample_rate,
                            challenge=self.challenge,
                            use_dataset_cache=self.use_dataset_cache_file,
                            raw_sample_filter=raw_sample_filter,
                            num_adj_slices=self.num_adj_slices,
                            data_balancer=data_balancer,
                        )

    def train_dataloader(self):
        return self._create_data_loader(self.slice_dataset, self.train_transform, data_partition="train")

    def val_dataloader(self):
        return self._create_data_loader(self.slice_dataset, self.val_transform, data_partition="val")
   
    def predict_dataloader(self):
        return self._create_data_loader(self.slice_dataset, self.val_transform, data_partition="val")
