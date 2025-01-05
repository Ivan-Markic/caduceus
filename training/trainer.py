import wandb
import torch
from pathlib import Path
from typing import Dict, List
import numpy as np
from tqdm import tqdm
from utils.metrics import SegmentationMetrics, MultiClassDiceLoss

class Trainer:
    def __init__(self, 
                 model: torch.nn.Module,
                 optimizer: torch.optim.Optimizer,
                 device: str,
                 checkpoint_dir: Path,
                 num_classes: int = 3,
                 patience: int = 10):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.criterion = MultiClassDiceLoss(num_classes=num_classes)
        self.metrics = SegmentationMetrics(num_classes=num_classes)
        self.best_loss = float('inf')
        self.patience = patience
        self.counter = 0
        self.best_metrics = None
        self.early_stop = False
        
    def _collect_case_metrics(self, metrics_list: List[Dict]) -> Dict:
        """
        Organize metrics by case, averaging scores across all slices of the same case
        Args:
            metrics_list: List of metrics dictionaries, where each dictionary contains metrics for one or more slices
                         belonging to the same case_id that were present in the batch
        """
        metrics_dict = {}
        
        metrics_dict['background_dice'] = np.mean([metrics['background_dice'] for metrics in metrics_list])
        metrics_dict['kidney_dice'] = np.mean([metrics['kidney_dice'] for metrics in metrics_list])
        metrics_dict['tumor_dice'] = np.mean([metrics['tumor_dice'] for metrics in metrics_list])
        metrics_dict['mean_dice'] = np.mean([metrics['mean_dice'] for metrics in metrics_list])
        metrics_dict['mean_kt_dice'] = np.mean([metrics['mean_kt_dice'] for metrics in metrics_list])
        metrics_dict['background_iou'] = np.mean([metrics['background_iou'] for metrics in metrics_list])
        metrics_dict['kidney_iou'] = np.mean([metrics['kidney_iou'] for metrics in metrics_list])
        metrics_dict['tumor_iou'] = np.mean([metrics['tumor_iou'] for metrics in metrics_list])
        metrics_dict['mean_iou'] = np.mean([metrics['mean_iou'] for metrics in metrics_list])
        metrics_dict['mean_kt_iou'] = np.mean([metrics['mean_kt_iou'] for metrics in metrics_list])

        return metrics_dict

    def train_epoch(self, dataloader, epoch: int):
        self.model.train()
        epoch_losses = []
        all_metrics = {}
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1} Training')
        for batch in pbar:
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            batch_case_ids = batch['case_id']
            
            # Forward pass
            outputs = self.model(images)
            
            # Calculate loss
            loss = self.criterion(outputs, masks)
            
            # Store loss
            epoch_losses.append(loss.item())

            # Calculate metrics
            for case_id in batch_case_ids:
                # Get indices for this case in the current batch
                case_indices = [i for i, bid in enumerate(batch_case_ids) if bid == case_id]
                case_outputs = outputs[case_indices]
                case_masks = masks[case_indices]
                
                # Calculate metrics for this case's slices
                metrics = self.metrics(case_outputs, case_masks)
                
                # Store metrics for this case
                if case_id not in all_metrics:
                    all_metrics[case_id] = []
                all_metrics[case_id].append(metrics)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'dice_kt': f'{metrics["mean_kt_dice"]:.4f}'
            })

        per_case_metrics = {}
        for case_id, metrics_list in all_metrics.items():
            per_case_metrics[case_id] = self._collect_case_metrics(metrics_list)

        # our all_metrics is a dictionary with case_ids as keys and a list of metrics as values
        # we need to collect the metrics for each case and then average them

        # Calculate epoch metrics with epoch number
        epoch_metrics = {
            'epoch': epoch,
            'train/loss': np.mean(epoch_losses),
            'train/dice_background': np.mean([metrics['background_dice'] for metrics in per_case_metrics.values()]),
            'train/dice_kidney': np.mean([metrics['kidney_dice'] for metrics in per_case_metrics.values()]),
            'train/dice_tumor': np.mean([metrics['tumor_dice'] for metrics in per_case_metrics.values()]),
            'train/dice_mean': np.mean([metrics['mean_dice'] for metrics in per_case_metrics.values()]),
            'train/dice_mean_kt': np.mean([metrics['mean_kt_dice'] for metrics in per_case_metrics.values()]),
            'train/iou_background': np.mean([metrics['background_iou'] for metrics in per_case_metrics.values()]),
            'train/iou_kidney': np.mean([metrics['kidney_iou'] for metrics in per_case_metrics.values()]),
            'train/iou_tumor': np.mean([metrics['tumor_iou'] for metrics in per_case_metrics.values()]),
            'train/iou_mean': np.mean([metrics['mean_iou'] for metrics in per_case_metrics.values()]),
            'train/iou_mean_kt': np.mean([metrics['mean_kt_iou'] for metrics in per_case_metrics.values()])
        }
        
        # Log metrics to wandb
        wandb.log(epoch_metrics)

        dice_table_data = []
        iou_table_data = []
        for case_id, metrics in per_case_metrics.items():
            dice_table_data.append([
                case_id, 
                metrics['background_dice'], 
                metrics['kidney_dice'], 
                metrics['tumor_dice'],
                metrics['mean_kt_dice'],
            ])

            iou_table_data.append([
                case_id, 
                metrics['background_iou'], 
                metrics['kidney_iou'], 
                metrics['tumor_iou'],
                metrics['mean_kt_iou'],
            ])

        # Log tables with epoch information
        wandb.log({
            f"train/Dice Per Case Epoch {epoch}": wandb.Table(
                data=dice_table_data,
                columns=['Case ID', 'Background', 'Kidney', 'Tumor', 'Mean KT']
            ),
            f"train/IoU Per Case Epoch {epoch}": wandb.Table(
                data=iou_table_data,
                columns=['Case ID', 'Background', 'Kidney', 'Tumor', 'Mean KT']
            ),
            'epoch': epoch
        })
        
        return epoch_metrics

    def validate(self, dataloader, epoch: int):
        self.model.eval()
        val_losses = []
        all_metrics = {}  # Dictionary to store metrics per case
        
        pbar = tqdm(dataloader, desc=f'Epoch {epoch+1} Validation')
        with torch.no_grad():
            for batch in pbar:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                batch_case_ids = batch['case_id']
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                val_losses.append(loss.item())

                
                # Calculate metrics per case in batch
                for case_id in batch_case_ids:
                    # Get indices for this case in the current batch
                    case_indices = [i for i, bid in enumerate(batch_case_ids) if bid == case_id]
                    case_outputs = outputs[case_indices]
                    case_masks = masks[case_indices]
                    
                    # Calculate metrics for this case's slices
                    metrics = self.metrics(case_outputs, case_masks)

                    # Store metrics for this case
                    if case_id not in all_metrics:
                        all_metrics[case_id] = []
                    all_metrics[case_id].append(metrics)
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'dice_kt': f'{metrics["mean_kt_dice"]:.4f}'
                })

        # Calculate per-case metrics
        per_case_metrics = {}
        for case_id, metrics_list in all_metrics.items():
            per_case_metrics[case_id] = self._collect_case_metrics(metrics_list)

        # Calculate validation metrics
        val_metrics = {
            'epoch': epoch,
            'val/loss': np.mean(val_losses),
            'val/dice_background': np.mean([metrics['background_dice'] for metrics in per_case_metrics.values()]),
            'val/dice_kidney': np.mean([metrics['kidney_dice'] for metrics in per_case_metrics.values()]),
            'val/dice_tumor': np.mean([metrics['tumor_dice'] for metrics in per_case_metrics.values()]),
            'val/dice_mean': np.mean([metrics['mean_dice'] for metrics in per_case_metrics.values()]),
            'val/dice_mean_kt': np.mean([metrics['mean_kt_dice'] for metrics in per_case_metrics.values()]),
            'val/iou_background': np.mean([metrics['background_iou'] for metrics in per_case_metrics.values()]),
            'val/iou_kidney': np.mean([metrics['kidney_iou'] for metrics in per_case_metrics.values()]),
            'val/iou_tumor': np.mean([metrics['tumor_iou'] for metrics in per_case_metrics.values()]),
            'val/iou_mean': np.mean([metrics['mean_iou'] for metrics in per_case_metrics.values()]),
            'val/iou_mean_kt': np.mean([metrics['mean_kt_iou'] for metrics in per_case_metrics.values()])
        }

        # Log metrics to wandb
        wandb.log(val_metrics)

        # Create tables for wandb
        dice_table_data = []
        iou_table_data = []
        
        for case_id, metrics in per_case_metrics.items():
            dice_table_data.append([
                case_id,
                metrics['background_dice'],
                metrics['kidney_dice'],
                metrics['tumor_dice'],
                metrics['mean_kt_dice']
            ])
            
            iou_table_data.append([
                case_id,
                metrics['background_iou'],
                metrics['kidney_iou'],
                metrics['tumor_iou'],
                metrics['mean_kt_iou']
            ])


        # Log tables to wandb
        wandb.log({
            f"val/Dice Per Case Epoch {epoch}": wandb.Table(
                data=dice_table_data,
                columns=['Case ID', 'Background', 'Kidney', 'Tumor', 'Mean KT']
            ),
            f"val/IoU Per Case Epoch {epoch}": wandb.Table(
                data=iou_table_data,
                columns=['Case ID', 'Background', 'Kidney', 'Tumor', 'Mean KT']
            ),
            'epoch': epoch
        })
        return val_metrics

    def train(self, train_loader, val_loader, num_epochs):
        """Main training loop with early stopping"""
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Training phase
            train_metrics = self.train_epoch(train_loader, epoch)
            print(f"Epoch {epoch+1} Training - Loss: {train_metrics['train/loss']:.4f}, "
                  f"Dice KT: {train_metrics['train/dice_mean_kt']:.4f}")
            
            # Validation phase
            val_metrics = self.validate(val_loader, epoch)
            print(f"Epoch {epoch+1} Validation - Loss: {val_metrics['val/loss']:.4f}, "
                  f"Dice KT: {val_metrics['val/dice_mean_kt']:.4f}")
            
            # Early stopping check
            if val_metrics['val/loss'] < self.best_loss or self.best_metrics is None:
                self.best_loss = val_metrics['val/loss']
                self.best_metrics = val_metrics
                self.counter = 0
                
                # Save best model
                self._save_checkpoint(epoch, val_metrics)
                print(f"New best model saved! (Loss: {self.best_loss:.4f})")
            else:
                self.counter += 1
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
                
                if self.counter >= self.patience:
                    print("Early stopping triggered!")
                    self.early_stop = True
                    break
        
        return self.best_metrics

    def _save_checkpoint(self, epoch, metrics):
        """Save model checkpoint"""
        checkpoint_path = str(self.checkpoint_dir / f'model_epoch_{epoch}_loss_{metrics["val/loss"]:.4f}.pth')
        
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': metrics["val/loss"],
            'metrics': metrics
        }, checkpoint_path)
        
        # Log to wandb
        artifact = wandb.Artifact(
            name=f'model-epoch-{epoch}',
            type='model',
            description=f'Model checkpoint from epoch {epoch} with validation loss {metrics["val/loss"]:.4f}'
        )
        artifact.add_file(checkpoint_path)
        wandb.log_artifact(artifact)
