# Sensitivity Maps Logging During Inference

This guide explains how to log sensitivity maps to wandb during the inference stage (predict mode) in the PromptMR-plus-Task4_debug repository.

## Overview

Sensitivity maps are crucial for understanding how the model processes multi-coil MRI data. This implementation allows you to visualize sensitivity maps alongside reconstructions during inference, helping you analyze the model's behavior on new data.

## What Was Modified

### 1. Updated `predict_step` Methods

Both `promptmr_module.py` and `parallel_promptmr_module.py` now return sensitivity maps and zero-filled images in their `predict_step` methods:

```python
def predict_step(self, batch, batch_idx, dataloader_idx=0):
    # ... existing code ...
    sense_map = output_dict['sens_maps'][:,0].abs()
    return {
        'output': output.cpu(), 
        'img_zf': img_zf.cpu(),  # Added zero-filled images
        'slice_num': batch.slice_num, 
        'fname': batch.fname,
        'num_slc': num_slc,
        'batch_idx': batch_idx,
        'time_frame': batch.num_t,
        'has_fake_time_dim': batch.has_fake_time_dim,
        'sens_maps': sense_map.cpu()  # Added sensitivity maps
    }
```

### 2. New Callback: `InferenceSensMapsLogger`

Created `callbacks/inference_sens_maps_logger.py` which:
- Logs sensitivity maps, zero-filled images, and reconstructions to wandb during inference
- **Follows the exact same logging pattern as the validation stage**
- Uses the same alpha adjustment (α=0.2) for brightness control
- Configurable logging frequency
- Handles batch processing
- Includes proper error handling

## How to Use

### Method 1: Using Configuration File

1. **Create or modify your inference config** to include the callback:

```yaml
trainer:
  logger:
    class_path: lightning.pytorch.loggers.WandbLogger
    init_args:
      project: "your-project-name"
      name: "inference-with-sensmaps"
      save_dir: "./wandb_logs"
  
  callbacks:
    - class_path: callbacks.inference_sens_maps_logger.InferenceSensMapsLogger
      init_args:
        log_every_n_batches: 10  # Log every 10 batches
        log_indices: [0, 5, 10]  # Also log specific batch indices
        enable_wandb_logging: true
```

2. **Run inference**:

```bash
python main.py predict \
    --config your_inference_config.yaml \
    --predict.ckpt_path path/to/checkpoint.ckpt
```

### Method 2: Using Command Line

```bash
python main.py predict \
    --config configs/inference/example_with_sensmaps.yaml \
    --predict.ckpt_path path/to/checkpoint.ckpt \
    --trainer.logger.init_args.project "my-inference-project" \
    --trainer.logger.init_args.name "inference_$(date +%Y%m%d_%H%M%S)"
```

### Method 3: Using the Provided Script

```bash
./scripts/run_inference_with_sensmaps.sh path/to/checkpoint.ckpt configs/inference/example_with_sensmaps.yaml
```

## Configuration Options

The `InferenceSensMapsLogger` callback accepts these parameters:

- `log_every_n_batches` (int, default: 10): Log every N batches
- `log_indices` (List[int], optional): Specific batch indices to log
- `enable_wandb_logging` (bool, default: True): Enable/disable wandb logging

## What You'll See in Wandb

In your wandb dashboard, you'll find images organized as:
- `inference/batch_000_item_0` - Sensitivity maps, zero-filled images, and reconstructions
- `inference/batch_010_item_0` - More examples
- Each image includes captions with filename and slice number

**The logging now follows the exact same pattern as validation stage:**
- **Sensitivity Maps**: Visual representation of coil sensitivity patterns (normalized)
- **Zero-filled Images**: Raw reconstruction from undersampled data (with α=0.2 brightness adjustment)
- **Reconstructions**: The model's output images (with α=0.2 brightness adjustment)

This matches the validation logging pattern: `[mask, sens_maps, img_zf**alpha, output**alpha, target**alpha, error]` but for inference we log: `[sens_maps, img_zf**alpha, output**alpha]` since we don't have targets or error maps during inference.

## Troubleshooting

### Common Issues

1. **"WandbLogger not found"**: Make sure you have a WandbLogger configured in your trainer
2. **"No 'sens_maps' found in outputs"**: Ensure you're using the updated predict_step methods
3. **Memory issues**: Reduce `log_every_n_batches` or use specific `log_indices` to log fewer images

### Debug Mode

To see what's happening, check the console output for messages like:
```
[InferenceSensMapsLogger] WandbLogger found. Sensitivity maps will be logged during inference.
[InferenceSensMapsLogger] Logged sensitivity map for batch 0, item 0: filename.mat
```

## Performance Considerations

- Logging every batch can slow down inference significantly
- Consider using `log_every_n_batches: 50` or higher for large datasets
- Use `log_indices` to log only specific interesting cases
- Sensitivity maps are normalized for visualization (divided by max value)

## Example Use Cases

1. **Model Analysis**: Understand how sensitivity maps vary across different anatomical regions
2. **Quality Control**: Verify that sensitivity maps look reasonable for new data
3. **Research**: Analyze the relationship between sensitivity patterns and reconstruction quality
4. **Debugging**: Identify cases where sensitivity estimation might be problematic

## Files Modified/Created

- `pl_modules/promptmr_module.py` - Added sens_maps to predict_step return
- `pl_modules/parallel_promptmr_module.py` - Added sens_maps to predict_step return  
- `callbacks/inference_sens_maps_logger.py` - New callback for logging
- `configs/inference/example_with_sensmaps.yaml` - Example configuration
- `scripts/run_inference_with_sensmaps.sh` - Example script
- `INFERENCE_SENSMAPS_README.md` - This documentation

## Next Steps

1. Test the implementation with your checkpoint
2. Adjust logging frequency based on your needs
3. Explore the sensitivity maps in wandb to gain insights
4. Consider adding additional visualizations (e.g., error maps, coil-specific sensitivity patterns)
