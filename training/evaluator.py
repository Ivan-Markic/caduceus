import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
from torch.utils.data import DataLoader
import nibabel as nib
from utils.metrics import MultiClassDiceScore

class Evaluator:
    def __init__(self, model, device, output_dir):
        self.model = model
        self.device = device
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.metric = MultiClassDiceScore()
        
    @torch.no_grad()
    def predict_batch(self, images):
        self.model.eval()
        images = images.to(self.device)
        outputs = self.model(images)
        predictions = torch.softmax(outputs, dim=1)
        return predictions.cpu()
    
    def predict_case(self, dataloader, case_id):
        """Predict masks for all slices in a case and reconstruct 3D volume."""
        predictions = []
        case_metrics = {
            'background': [],
            'kidney': [],
            'tumor': [],
            'mean': []
        }
        
        # Get affine from first batch
        affine = None
        
        for batch in tqdm(dataloader, desc=f"Predicting case {case_id}"):
            if affine is None:
                affine = batch['affine'][0]  # Get affine from first batch
                
            batch_preds = self.predict_batch(batch['image'])
            
            # Calculate metrics if masks are available
            if 'mask' in batch:
                dice_scores = self.metric(batch_preds, batch['mask'])
                for key in case_metrics:
                    case_metrics[key].append(dice_scores[key])
            
            # Convert predictions to class indices
            batch_preds = torch.argmax(batch_preds, dim=1).numpy()
            predictions.extend([p for p in batch_preds])
            
        # Stack predictions into 3D volume
        volume = np.stack(predictions, axis=0)
        
        # Create NIfTI image using affine
        nifti_img = nib.Nifti1Image(volume.astype(np.uint8), affine)
        
        # Save prediction
        output_path = self.output_dir / f"{case_id}_prediction.nii.gz"
        nib.save(nifti_img, output_path)
        
        # Calculate mean metrics for the case
        case_summary = {}
        if case_metrics['mean']:
            for key in case_metrics:
                case_summary[key] = np.mean(case_metrics[key])
        
        return output_path, case_summary
