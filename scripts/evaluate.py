import click
import torch
from pathlib import Path
import json
from torch.utils.data import DataLoader
import numpy as np
import nibabel as nib

from models.unet import UNet
from dataset.dataset import CTDataset
from preprocessing.transforms import get_validation_transforms
from training.evaluator import Evaluator

@click.command()
@click.option('--data-dir', type=click.Path(exists=True), required=True,
              help='Path to preprocessed dataset directory')
@click.option('--model-path', type=click.Path(exists=True), required=True,
              help='Path to trained model checkpoint')
@click.option('--output-dir', type=click.Path(), required=True,
              help='Directory to save evaluation results')
@click.option('--split', type=str, default='test',
              help='Dataset split to evaluate on (train/test)')
@click.option('--batch-size', type=int, default=4,
              help='Batch size for evaluation')
@click.option('--device', type=str, default='cuda',
              help='Device to use for inference')
def main(data_dir, model_path, output_dir, split, batch_size, device):
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = UNet(in_channels=1, out_channels=3)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    # Create evaluator
    evaluator = Evaluator(model, device, output_dir)
    
    # Create dataset and dataloader
    dataset = CTDataset(
        data_dir=data_dir,
        split=split,
        transform=get_validation_transforms()
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,  # Use the provided batch_size
        shuffle=False,
        num_workers=4
    )
    
    # Evaluate
    print(f"Evaluating model on {split} set...")
    results = {}
    
    # Group data by case
    current_case = None
    case_predictions = []
    case_targets = []
    first_affine = None
    
    for batch in dataloader:
        # Get unique case IDs in the batch
        case_ids = batch['case_id']
        unique_cases = torch.unique(case_ids)
        
        # Process each case in the batch
        for case_id in unique_cases:
            case_mask = case_ids == case_id
            case_images = batch['image'][case_mask]
            case_masks = batch['mask'][case_mask] if 'mask' in batch else None
            case_affine = batch['affine'][case_mask][0]  # Take first affine matrix for the case
            
            if current_case is None:
                current_case = case_id
                first_affine = case_affine
            
            if case_id != current_case:
                # Process previous case
                predictions = torch.cat(case_predictions, dim=0)
                targets = torch.cat(case_targets, dim=0) if case_targets else None
                
                # Save predictions and calculate metrics
                pred_volume = torch.argmax(predictions, dim=1).cpu().numpy()
                nifti_img = nib.Nifti1Image(pred_volume.astype(np.uint8), first_affine.numpy())
                pred_path = output_dir / f"{current_case}_prediction.nii.gz"
                nib.save(nifti_img, pred_path)
                
                # Calculate metrics if we have targets
                if targets is not None:
                    scores = evaluator.metric(predictions, targets)
                    results[current_case] = {
                        'background': scores['background'],
                        'kidney': scores['kidney'],
                        'tumor': scores['tumor'],
                        'mean': scores['mean']
                    }
                
                # Start new case
                current_case = case_id
                case_predictions = []
                case_targets = []
                first_affine = case_affine
            
            # Process current batch
            with torch.no_grad():
                output = model(case_images.to(device))
                pred = torch.softmax(output, dim=1)
                case_predictions.append(pred.cpu())
                
                if case_masks is not None:
                    case_targets.append(case_masks.cpu())
    
    # Process last case
    if case_predictions:
        predictions = torch.cat(case_predictions, dim=0)
        targets = torch.cat(case_targets, dim=0) if case_targets else None
        
        pred_volume = torch.argmax(predictions, dim=1).cpu().numpy()
        nifti_img = nib.Nifti1Image(pred_volume.astype(np.uint8), first_affine.numpy())
        pred_path = output_dir / f"{current_case}_prediction.nii.gz"
        nib.save(nifti_img, pred_path)
        
        if targets is not None:
            scores = evaluator.metric(predictions, targets)
            results[current_case] = {
                'background': scores['background'],
                'kidney': scores['kidney'],
                'tumor': scores['tumor'],
                'mean': scores['mean']
            }
    
    # Calculate overall means
    if results:
        overall_means = {
            'background': sum(case['background'] for case in results.values()) / len(results),
            'kidney': sum(case['kidney'] for case in results.values()) / len(results),
            'tumor': sum(case['tumor'] for case in results.values()) / len(results),
            'mean': sum(case['mean'] for case in results.values()) / len(results)
        }
        
        results['overall'] = overall_means
        
        # Save results
        output_path = output_dir / f'{split}_evaluation_results.json'
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print("\nOverall Results:")
        print(f"Background Dice: {overall_means['background']:.4f}")
        print(f"Kidney Dice: {overall_means['kidney']:.4f}")
        print(f"Tumor Dice: {overall_means['tumor']:.4f}")
        print(f"Mean Dice: {overall_means['mean']:.4f}")
        print(f"\nResults saved to {output_path}")

if __name__ == '__main__':
    main() 