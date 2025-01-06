import click
import torch
import wandb
from pathlib import Path
from torch.utils.data import DataLoader

from models.unet import UNet
from dataset.dataset import CTDataset
from preprocessing.transforms import get_validation_transforms
from training.trainer import Trainer

@click.command()
@click.option('--data-dir', type=click.Path(exists=True), required=True,
              help='Path to preprocessed dataset directory')
@click.option('--model-path', type=click.Path(exists=True), required=True,
              help='Path to best model checkpoint')
@click.option('--batch-size', type=int, default=4,
              help='Batch size for evaluation')
@click.option('--device', type=str, default='cuda',
              help='Device to use for evaluation')
@click.option('--checkpoint-dir', type=click.Path(exists=True), default='checkpoints',
              help='Path to checkpoint directory')
@click.option('--wandb-project', type=str, default='from_scratch_unet',
              help='WandB project name')
def evaluate(data_dir, model_path, batch_size, device, wandb_project, checkpoint_dir):
    # Initialize wandb
    wandb.init(
        project=wandb_project,
        name='best-unet-eval',
        config={
            "batch_size": batch_size,
            "architecture": "UNet",
            "dataset": "KiTS19",
            "device": device
        }
    )
    
    # Load model
    model = UNet(in_channels=1, out_channels=3)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Create dataset for all cases (0-210)
    dataset = CTDataset(
        data_dir=data_dir,
        split='eval',  # This will load all cases
        transform=get_validation_transforms()
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create trainer instance
    trainer = Trainer(
        model=model,
        optimizer=None,  # Not needed for evaluation
        device=device,
        checkpoint_dir=checkpoint_dir,  # Not needed for evaluation
        patience=None  # Not needed for evaluation
    )
    
    # Run validation on all cases
    print("Evaluating model on all cases...")
    metrics = trainer.validate(dataloader, epoch=0)
    
    print("\nEvaluation complete!")
    print(f"Overall Mean KT Dice: {metrics['val/dice_mean_kt']:.4f}")
    
    wandb.finish()

if __name__ == '__main__':
    evaluate() 