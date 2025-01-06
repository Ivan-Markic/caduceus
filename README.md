# Kidney Tumor Segmentation Project

This project implements a UNet model for kidney and tumor segmentation using the KiTS19 dataset.

## Setup and Training Pipeline

### Prerequisites
First, install all required dependencies:

```bash
pip install -r requirements.txt
```

Required packages:
- Deep Learning Framework
    - torch==2.1.2
    - torchvision==0.16.2

- Medical Image Processing
    - nibabel==5.2.0

- Image Processing and Data Manipulation
    - numpy==1.24.3
    - opencv-python==4.9.0.80
    - albumentations==1.3.1

- Progress Bars and CLI
    - tqdm==4.66.1
    - click==8.1.7

- Logging
    - wandb==0.15.11

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

### 3. Model Evaluation
After training, evaluate the model's performance on all cases:

```bash
python -m scripts.evaluate_model \
    --data-dir preprocessed_data \
    --model-path checkpoints/unet/model_epoch_19_loss_0.5370.pth \
    --batch-size 12 \
    --device cuda
```

Evaluation features:
- Comprehensive metrics for all cases (0-210)
- Per-case Dice and IoU scores
- WandB logging with detailed tables
- Validation metrics for kidney and tumor segmentation
- Uses same validation pipeline as training

## Model Architecture
- UNet with 1 input channel and 3 output channels (background, kidney, tumor)
- Trained using Adam optimizer
- Dice loss for segmentation optimization
