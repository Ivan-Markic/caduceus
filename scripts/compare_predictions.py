import click
import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib
from pathlib import Path
from utils.metrics import MultiClassDiceLoss, MultiClassDiceScore

def print_class_details(pred_tensor, gt_tensor, class_idx, class_name):
    """Debug function to print detailed metrics for a specific class"""
    pred_class = (pred_tensor == class_idx)
    gt_class = (gt_tensor == class_idx)
    
    intersection = (pred_class & gt_class).sum().item()
    pred_sum = pred_class.sum().item()
    gt_sum = gt_class.sum().item()
    
    print(f"\nDetailed {class_name} metrics:")
    print(f"Predicted pixels: {pred_sum}")
    print(f"Ground truth pixels: {gt_sum}")
    print(f"Intersection pixels: {intersection}")
    if pred_sum + gt_sum > 0:
        dice = (2.0 * intersection) / (pred_sum + gt_sum)
        print(f"Calculated Dice score: {dice:.4f}")

@click.command()
@click.option('--data-dir', type=click.Path(exists=True), required=True,
              help='Path to preprocessed dataset directory')
@click.option('--pred-path', type=click.Path(exists=True), required=True,
              help='Path to prediction .nii.gz file')
@click.option('--case-id', type=str, default='case_00000',
              help='Case ID to compare')
def main(data_dir, pred_path, case_id):
    # Load prediction
    pred_nii = nib.load(pred_path)
    pred = pred_nii.get_fdata()
    
    # Load ground truth masks
    case_dir = Path(data_dir) / case_id / 'masks'
    gt_slices = []
    for mask_path in sorted(case_dir.glob('slice_*.npy')):
        gt_slice = np.load(mask_path)
        gt_slices.append(gt_slice)
    gt = np.stack(gt_slices)
    
    # Convert to tensors (keeping all slices)
    pred_tensor = torch.from_numpy(pred).long()
    gt_tensor = torch.from_numpy(gt).long()
    
    # Print class distribution first
    print("\nClass distribution in prediction:")
    for i in range(3):
        pixels = (pred_tensor == i).sum().item()
        percentage = (pixels / pred_tensor.numel()) * 100
        print(f"Class {i}: {pixels} pixels ({percentage:.2f}%)")
    
    print("\nClass distribution in ground truth:")
    for i in range(3):
        pixels = (gt_tensor == i).sum().item()
        percentage = (pixels / gt_tensor.numel()) * 100
        print(f"Class {i}: {pixels} pixels ({percentage:.2f}%)")
    
    # Print detailed metrics for each class
    class_names = ['Background', 'Kidney', 'Tumor']
    for i, name in enumerate(class_names):
        print_class_details(pred_tensor, gt_tensor, i, name)
    
    # Calculate metrics for full volume
    pred_tensor_input = F.one_hot(pred_tensor, num_classes=3).permute(0, 3, 1, 2).float()
    gt_tensor_input = gt_tensor
    
    criterion = MultiClassDiceLoss()
    dice_score = MultiClassDiceScore()
    
    loss = criterion(pred_tensor_input, gt_tensor_input)
    scores = dice_score(pred_tensor_input, gt_tensor_input)
    
    print(f"\nMetrics for full volume:")
    print(f"Loss: {loss.item():.4f}")
    print("\nDice Scores:")
    print(f"Background: {scores['background']:.4f}")
    print(f"Kidney: {scores['kidney']:.4f}")
    print(f"Tumor: {scores['tumor']:.4f}")
    print(f"Mean: {scores['mean']:.4f}")

if __name__ == '__main__':
    main() 