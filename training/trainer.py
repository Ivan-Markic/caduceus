import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import json
from tqdm import tqdm
import logging
from typing import Dict, Optional
import numpy as np

from utils.metrics import MultiClassDiceScore

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        num_epochs: int = 100,
        patience: int = 10,
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs"
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.scheduler = scheduler
        self.num_epochs = num_epochs
        self.patience = patience
        
        # Create directories
        self.checkpoint_dir = Path(checkpoint_dir)
        self.log_dir = Path(log_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.log_dir.mkdir(exist_ok=True)
        
        self.dice_metric = MultiClassDiceScore()
        self.best_mean_dice = 0.0
        self.epochs_without_improvement = 0
        
    def _log_case_metrics(self, case_id: str, metrics: Dict, phase: str, epoch: int):
        """Log metrics for a specific case."""
        log_file = self.log_dir / f"{case_id}_{phase}_metrics.json"
        
        # Load existing metrics if file exists
        if log_file.exists():
            with open(log_file, 'r') as f:
                case_history = json.load(f)
        else:
            case_history = {}
            
        # Add new metrics
        case_history[f"epoch_{epoch}"] = metrics
        
        # Save updated metrics
        with open(log_file, 'w') as f:
            json.dump(case_history, f, indent=2)
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.train()
        epoch_loss = 0
        epoch_metrics = {
            'background': [], 'kidney': [], 'tumor': [], 'mean': []
        }
        num_batches = len(self.train_loader)
        
        with tqdm(self.train_loader, desc=f"Training Epoch {epoch}") as pbar:
            for batch in pbar:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                case_ids = batch['case_id']  # Added case_id to dataset returns
                
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                
                loss.backward()
                self.optimizer.step()
                
                # Calculate metrics
                dice_scores = self.dice_metric(outputs, masks)
                
                # Update metrics
                epoch_loss += loss.item()
                for key in epoch_metrics:
                    epoch_metrics[key].append(dice_scores[key])
                
                # Log metrics for each case in the batch
                for idx, case_id in enumerate(case_ids):
                    case_metrics = {
                        'loss': loss.item(),
                        'background_dice': dice_scores['background'],
                        'kidney_dice': dice_scores['kidney'],
                        'tumor_dice': dice_scores['tumor'],
                        'mean_dice': dice_scores['mean']
                    }
                    self._log_case_metrics(case_id, case_metrics, 'train', epoch)
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'mean_dice': f'{dice_scores["mean"]:.4f}'
                })
        
        # Calculate epoch averages
        return {
            'loss': epoch_loss / num_batches,
            'background_dice': np.mean(epoch_metrics['background']),
            'kidney_dice': np.mean(epoch_metrics['kidney']),
            'tumor_dice': np.mean(epoch_metrics['tumor']),
            'mean_dice': np.mean(epoch_metrics['mean'])
        }
    
    @torch.no_grad()
    def validate_epoch(self, epoch: int) -> Dict[str, float]:
        self.model.eval()
        epoch_loss = 0
        epoch_metrics = {
            'background': [], 'kidney': [], 'tumor': [], 'mean': []
        }
        num_batches = len(self.val_loader)
        
        with tqdm(self.val_loader, desc="Validation") as pbar:
            for batch in pbar:
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                case_ids = batch['case_id']
                
                outputs = self.model(images)
                loss = self.criterion(outputs, masks)
                dice_scores = self.dice_metric(outputs, masks)
                
                # Update metrics
                epoch_loss += loss.item()
                for key in epoch_metrics:
                    epoch_metrics[key].append(dice_scores[key])
                
                # Log metrics for each case
                for idx, case_id in enumerate(case_ids):
                    case_metrics = {
                        'loss': loss.item(),
                        'background_dice': dice_scores['background'],
                        'kidney_dice': dice_scores['kidney'],
                        'tumor_dice': dice_scores['tumor'],
                        'mean_dice': dice_scores['mean']
                    }
                    self._log_case_metrics(case_id, case_metrics, 'val', epoch)
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'mean_dice': f'{dice_scores["mean"]:.4f}'
                })
        
        return {
            'loss': epoch_loss / num_batches,
            'background_dice': np.mean(epoch_metrics['background']),
            'kidney_dice': np.mean(epoch_metrics['kidney']),
            'tumor_dice': np.mean(epoch_metrics['tumor']),
            'mean_dice': np.mean(epoch_metrics['mean'])
        }
    
    def save_checkpoint(self, epoch: int, metrics: dict):
        """Save model checkpoint if validation performance improves."""
        current_dice = metrics['val_mean_dice']
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'metrics': metrics,
        }
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / 'latest_model.pth'
        torch.save(checkpoint, latest_path)
        
        # Save best model if performance improves
        if current_dice > self.best_mean_dice:
            self.best_mean_dice = current_dice
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"\nSaved best model with mean dice: {current_dice:.4f}")
    
    def train(self):
        try:
            for epoch in range(self.num_epochs):
                print(f"\nEpoch {epoch+1}/{self.num_epochs}")
                
                train_metrics = self.train_epoch(epoch)
                val_metrics = self.validate_epoch(epoch)
                
                metrics = {
                    'train_loss': train_metrics['loss'],
                    'train_background_dice': train_metrics['background_dice'],
                    'train_kidney_dice': train_metrics['kidney_dice'],
                    'train_tumor_dice': train_metrics['tumor_dice'],
                    'train_mean_dice': train_metrics['mean_dice'],
                    'val_loss': val_metrics['loss'],
                    'val_background_dice': val_metrics['background_dice'],
                    'val_kidney_dice': val_metrics['kidney_dice'],
                    'val_tumor_dice': val_metrics['tumor_dice'],
                    'val_mean_dice': val_metrics['mean_dice']
                }
                
                print(
                    f"Train - Loss: {metrics['train_loss']:.4f}, "
                    f"Background: {metrics['train_background_dice']:.4f}, "
                    f"Kidney: {metrics['train_kidney_dice']:.4f}, "
                    f"Tumor: {metrics['train_tumor_dice']:.4f}, "
                    f"Mean: {metrics['train_mean_dice']:.4f}"
                )
                print(
                    f"Val - Loss: {metrics['val_loss']:.4f}, "
                    f"Background: {metrics['val_background_dice']:.4f}, "
                    f"Kidney: {metrics['val_kidney_dice']:.4f}, "
                    f"Tumor: {metrics['val_tumor_dice']:.4f}, "
                    f"Mean: {metrics['val_mean_dice']:.4f}"
                )
                
                if self.scheduler is not None:
                    self.scheduler.step(metrics['val_loss'])
                
                self.save_checkpoint(epoch, metrics)
                
                # Early stopping based on mean dice
                if metrics['val_mean_dice'] > self.best_mean_dice:
                    self.best_mean_dice = metrics['val_mean_dice']
                    self.epochs_without_improvement = 0
                else:
                    self.epochs_without_improvement += 1
                    
                if self.epochs_without_improvement >= self.patience:
                    print(f"\nEarly stopping after {epoch+1} epochs")
                    break
                
        except KeyboardInterrupt:
            print("\nTraining interrupted. Saving current state...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            }, Path(self.checkpoint_dir) / 'interrupted.pth')
            print("Saved interrupted.pth")
            raise KeyboardInterrupt
