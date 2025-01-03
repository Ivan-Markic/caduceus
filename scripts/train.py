import click
import torch
import wandb
from torch.utils.data import DataLoader

from models.unet import UNet
from dataset.dataset import CTDataset
from preprocessing.transforms import get_training_transforms, get_validation_transforms
from training.trainer import Trainer

@click.command()
@click.option('--data-dir', type=click.Path(exists=True), required=True,
              help='Path to preprocessed dataset directory')
@click.option('--checkpoint-dir', type=click.Path(), required=True,
              help='Directory to save model checkpoints')
@click.option('--wandb-project', type=str, default='from_scratch_unet',
              help='WandB project name')
@click.option('--wandb-run-name', type=str, default='unet-training-run-',
              help='WandB run name (default: unet-training-run-{{epoch}})')
@click.option('--batch-size', type=int, default=4,
              help='Batch size for training')
@click.option('--num-epochs', type=int, default=100,
              help='Number of epochs to train')
@click.option('--learning-rate', type=float, default=1e-4,
              help='Learning rate')
@click.option('--device', type=str, default='cuda',
              help='Device to use for training (cuda/cpu)')
@click.option('--num-workers', type=int, default=0,
              help='Number of workers for data loading')
@click.option('--patience', type=int, default=10,
              help='Number of epochs to wait for improvement before early stopping')
def main(data_dir, checkpoint_dir, wandb_project, wandb_run_name, batch_size, 
         num_epochs, learning_rate, device, num_workers, patience):
    
    # Initialize wandb
    wandb.init(
        project=wandb_project,
        name=f'{wandb_run_name}{num_epochs}',
        config={
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "architecture": "UNet",
            "dataset": "KiTS19",
            "patience": patience,
            "device": device
        }
    )
    
    # Create datasets
    train_dataset = CTDataset(
        data_dir=data_dir,
        split='train',
        transform=get_training_transforms()
    )
    
    val_dataset = CTDataset(
        data_dir=data_dir,
        split='valid',
        transform=get_validation_transforms()
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Create model
    model = UNet(in_channels=1, out_channels=3)
    model = model.to(device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Create trainer with early stopping
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        device=device,
        checkpoint_dir=checkpoint_dir,
        patience=patience  # Use the patience from command line
    )
    
    # Training loop with early stopping
    print("Starting training...")
    best_metrics = trainer.train(train_loader, val_loader, num_epochs)
    
    print("\nTraining finished!")
    print(f"Best validation loss: {best_metrics['val/loss']:.4f}")
    print(f"Best validation Dice KT: {best_metrics['val/dice_mean_kt']:.4f}")
    
    wandb.finish()

if __name__ == '__main__':
    main()