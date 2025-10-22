#!/usr/bin/env python3
"""
Training script for multi-dataset (2024 + 2025) training
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import yaml
import argparse
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description='Start multi-dataset training')
    parser.add_argument('--config', type=str, 
                       default='configs/train/pmr-plus/cmr24-25-cardiac-task3.yaml',
                       help='Path to training configuration file')
    parser.add_argument('--dry-run', action='store_true',
                       help='Dry run to test configuration without starting training')
    
    args = parser.parse_args()
    
    config_path = Path(args.config)
    if not config_path.exists():
        logging.error(f"Configuration file not found: {config_path}")
        return
    
    logging.info(f"Loading configuration from: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    logging.info("Configuration loaded successfully!")
    
    # Print configuration summary
    data_config = config['data']['init_args']
    model_config = config['model']['init_args']
    trainer_config = config['trainer']
    
    logging.info("=" * 60)
    logging.info("TRAINING CONFIGURATION SUMMARY")
    logging.info("=" * 60)
    logging.info(f"Data Module: {config['data']['class_path']}")
    logging.info(f"Data Paths: {len(data_config['data_paths'])}")
    for i, path in enumerate(data_config['data_paths']):
        logging.info(f"  {i+1}. {path}")
    
    logging.info(f"Data Balancer: {data_config['data_balancer']['class_path']}")
    logging.info(f"Sequence Types: {len(data_config['data_balancer']['init_args']['ratio_dict'])}")
    
    logging.info(f"Model: {config['model']['class_path']}")
    logging.info(f"  Cascades: {model_config['num_cascades']}")
    logging.info(f"  Adjacent Slices: {model_config['num_adj_slices']}")
    logging.info(f"  Learning Rate: {model_config['lr']}")
    logging.info(f"  n_history: {model_config['n_history']}")
    
    logging.info(f"Trainer:")
    logging.info(f"  Devices: {trainer_config['devices']}")
    logging.info(f"  Max Epochs: {trainer_config['max_epochs']}")
    logging.info(f"  Project: {trainer_config['logger']['init_args']['project']}")
    logging.info(f"  Run Name: {trainer_config['logger']['init_args']['name']}")
    
    if args.dry_run:
        logging.info("DRY RUN MODE - Configuration validated successfully!")
        logging.info("To start actual training, run without --dry-run flag")
        return
    
    # Start actual training
    logging.info("=" * 60)
    logging.info("STARTING TRAINING")
    logging.info("=" * 60)
    
    try:
        # Import and run training
        from main import main as train_main
        
        # Set up arguments for main function
        import sys
        sys.argv = ['main.py', '--config', str(config_path)]
        
        train_main()
        
    except Exception as e:
        logging.error(f"Training failed with error: {e}")
        raise

if __name__ == "__main__":
    main()
