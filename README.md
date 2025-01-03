# Kidney Tumor Segmentation Project

This project implements a UNet model for kidney and tumor segmentation using the KiTS19 dataset.

## Setup and Training Pipeline

### 1. Data Preprocessing
First, run the data conversion script to preprocess the KiTS19 dataset:


```bash
python -m scripts.convert_data \
--data-dir ../kits19-challenge/kits19/data \
--output-dir preprocessed_data
```

This script:
- Converts .nii.gz files to .npy format
- Applies CT windowing
- Splits data into slices
- Saves images and corresponding masks
- Preserves affine transformations

### 2. Model Training
After preprocessing, train the UNet model:

```bash
python -m scripts.train \
--data-dir preprocessed_data \
--checkpoint-dir checkpoints/unet \
--batch-size 24 \
--num-epochs 20 \
--learning-rate 1e-4 \
--device cuda \
--num-workers 0 \
--patience 10
```

Training features:
- Early stopping with configurable patience
- Automatic model checkpointing
- Wandb integration for experiment tracking
- Dice loss and metrics for kidney and tumor segmentation
- Progress monitoring with validation metrics

## Model Architecture
- UNet with 1 input channel and 3 output channels (background, kidney, tumor)
- Trained using Adam optimizer
- Dice loss for segmentation optimization
