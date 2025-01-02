import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

class MultiClassDiceLoss(nn.Module):
    def __init__(self, num_classes: int = 3, smooth: float = 1e-6):
        super().__init__()
        self.num_classes = num_classes
        self.smooth = smooth
        
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Calculate Dice loss for multiple classes
        Args:
            pred: (B, C, H, W) logits
            target: (B, H, W) with class indices
        """
        target = target.squeeze(1)
        target_one_hot = F.one_hot(target, num_classes=self.num_classes)
        target_one_hot = target_one_hot.permute(0, 3, 1, 2).float()
        
        pred_softmax = F.softmax(pred, dim=1)
        
        total_loss = 0
        
        for class_idx in range(self.num_classes):
            pred_class = pred_softmax[:, class_idx]
            target_class = target_one_hot[:, class_idx]
            
            intersection = (pred_class * target_class).sum()
            union = pred_class.sum() + target_class.sum()
            
            # Only calculate loss if there are pixels of this class
            if union > 0:
                dice_score = (2.0 * intersection + self.smooth) / (union + self.smooth)
                dice_loss = 1 - dice_score
            else:
                dice_loss = torch.tensor(0.0, device=pred.device)
            
            total_loss += dice_loss
            
        return total_loss / self.num_classes

class MultiClassDiceScore:
    def __init__(self, num_classes: int = 3, smooth: float = 1e-6):
        self.num_classes = num_classes
        self.smooth = smooth
        self.class_names = ['background', 'kidney', 'tumor']
        
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """
        Calculate Dice score for each class
        Args:
            pred: (B, C, H, W) logits
            target: (B, 1, H, W) with class indices
        Returns:
            Dictionary with dice scores for each class
        """
        # Remove channel dimension from target
        target = target.squeeze(1)
        
        # Convert predictions to class indices
        pred = torch.softmax(pred, dim=1)
        pred = torch.argmax(pred, dim=1)
        
        dice_scores = {}
        
        for class_idx in range(self.num_classes):
            pred_class = (pred == class_idx).float()
            target_class = (target == class_idx).float()
            
            intersection = (pred_class * target_class).sum()
            union = pred_class.sum() + target_class.sum()
            
            # Only use smoothing if there are actual pixels of this class
            if union > 0:
                dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
            else:
                dice = torch.tensor(0.0)  # If no pixels of this class exist
            
            dice_scores[self.class_names[class_idx]] = dice.item()
        
        # Calculate mean dice
        dice_scores['mean'] = sum(dice_scores.values()) / self.num_classes
        return dice_scores
