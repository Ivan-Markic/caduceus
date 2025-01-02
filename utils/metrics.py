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

class SegmentationMetrics:
    def __init__(self, num_classes: int = 3, smooth: float = 1e-6):
        self.num_classes = num_classes
        self.smooth = smooth
        self.class_mapping = {
            0: 'background',
            1: 'kidney',
            2: 'tumor'
        }

    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
        """
        Calculate Dice score and IoU for each class
        Args:
            pred: (N, C, H, W) tensor of predictions (before softmax)
            target: (N, H, W) tensor of ground truth labels
        Returns:
            Dictionary containing dice scores and IoU for each class and means
        """
        pred = torch.softmax(pred, dim=1)
        pred = torch.argmax(pred, dim=1)
        
        scores = {}
        dice_scores = []
        iou_scores = []
        kt_dice_scores = []
        kt_iou_scores = []
        
        # Calculate scores for each class
        for i in range(self.num_classes):
            pred_class = (pred == i)
            target_class = (target == i)
            
            # Calculate Dice score
            dice = self._dice_score(pred_class, target_class)
            scores[f'{self.class_mapping[i]}_dice'] = dice
            dice_scores.append(dice)
            
            # Calculate IoU score
            iou = self._iou_score(pred_class, target_class)
            scores[f'{self.class_mapping[i]}_iou'] = iou
            iou_scores.append(iou)
            
            # Collect kidney and tumor scores
            if i > 0:  # Skip background
                kt_dice_scores.append(dice)
                kt_iou_scores.append(iou)
        
        # Calculate means
        scores['mean_dice'] = sum(dice_scores) / len(dice_scores)
        scores['mean_iou'] = sum(iou_scores) / len(iou_scores)
        scores['mean_kt_dice'] = sum(kt_dice_scores) / len(kt_dice_scores)
        scores['mean_kt_iou'] = sum(kt_iou_scores) / len(kt_iou_scores)
        
        return scores
    
    def _dice_score(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate Dice score for a single class"""
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        
        if union > 0:
            dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        else:
            dice = torch.tensor(1.0, device=pred.device)  # Perfect score if both are empty
            
        return dice.item()
    
    def _iou_score(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        """Calculate IoU score for a single class"""
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection
        
        if union > 0:
            iou = (intersection + self.smooth) / (union + self.smooth)
        else:
            iou = torch.tensor(1.0, device=pred.device)  # Perfect score if both are empty
            
        return iou.item()
