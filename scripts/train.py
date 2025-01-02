import click
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from dataset import CTDataset
from preprocessing.transforms import get_training_transforms, get_validation_transforms
from models.unet import UNet
from utils.metrics import MultiClassDiceLoss
from training.trainer import Trainer

@click.command()
@click.option('--data-dir', type=click.Path(exists=True), required=True,
              help='Path to preprocessed dataset directory')
@click.option('--batch-size', type=int, default=8, help='Batch size')
@click.option('--num-workers', type=int, default=4, help='Number of data loading workers')
@click.option('--learning-rate', type=float, default=1e-4, help='Learning rate')
@click.option('--num-epochs', type=int, default=100, help='Number of epochs')
@click.option('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
@click.option('--checkpoint-dir', type=click.Path(), default='checkpoints',
              help='Directory to save model checkpoints')
@click.option('--log-dir', type=click.Path(), default='logs',
              help='Directory to save training logs')
def main(data_dir, batch_size, num_workers, learning_rate, num_epochs, 
         device, checkpoint_dir, log_dir):
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
    
    # Create optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = MultiClassDiceLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.1,
        patience=5,
        verbose=True
    )
    
    # Create trainer with checkpoint and log directories
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=num_epochs,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir
    )
    
    trainer.train()

if __name__ == '__main__':
    main()