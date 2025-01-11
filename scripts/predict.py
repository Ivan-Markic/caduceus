import click
import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm

from models.unet import UNet
from dataset.dataset import CTDataset
from preprocessing.transforms import get_validation_transforms

@click.command()
@click.option('--data-dir', type=click.Path(exists=True), default='preprocessed_data',
              help='Path to preprocessed dataset directory')
@click.option('--model-path', type=click.Path(exists=True), default='checkpoints/unet/model_epoch_19_loss_0.5370.pth',
              help='Path to trained model checkpoint')
@click.option('--output-dir', type=click.Path(), default='../kits19-challenge/kits19',
              help='Directory to save predicted masks')
@click.option('--device', type=str, default='cuda',
              help='Device to use for inference')
def main(data_dir, model_path, output_dir, device):
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    model = UNet(in_channels=1, out_channels=3)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Create dataset for single case
    dataset = CTDataset(
        data_dir=data_dir,
        split='predict',  # Use test split for prediction
        transform=get_validation_transforms(),
    )

    # Process each case separately
    for case_id in dataset.case_ids:
        n_slices = dataset.get_case_slices(case_id)
        predictions = np.zeros((n_slices, 512, 512), dtype=np.uint8)
        
        print(f"Processing case {case_id}")
        with torch.no_grad():
            for slice_idx in tqdm(range(n_slices)):
                batch = dataset.get_case_data(case_id, slice_idx)
                image = batch['image'].unsqueeze(0).to(device)
                
                output = model(image)
                pred = torch.softmax(output, dim=1)
                pred = torch.argmax(pred, dim=1)
                pred = pred.cpu().numpy()[0]
                
                predictions[slice_idx] = pred
        
        # Create NIFTI image and save
        nifti_img = nib.Nifti1Image(predictions, batch['affine'])
        nifiti_img_filename = output_dir / f'{case_id}' / f'{case_id}_prediction.nii.gz'
        nifiti_img_filename.parent.mkdir(parents=True, exist_ok=True)
        
        nifti_img.to_filename(str(nifiti_img_filename))
        print(f"Saved predicted mask to {nifiti_img_filename}")

if __name__ == '__main__':
    main() 