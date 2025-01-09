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

    for index, case_id in enumerate(dataset.case_ids):
        # Get affine matrix
        affine = dataset.get_affine(case_id)

        # Initialize 3D array for predicted mask
        first_slice = np.load(dataset.image_paths[index])

        pred_shape = (len(dataset), *first_slice.shape)
        predictions = np.zeros(pred_shape, dtype=np.uint8)
        
        # Generate predicted mask
        print(f"Generating predicted mask for {case_id}")
        with torch.no_grad():
            for idx in tqdm(range(len(dataset))):
                batch = dataset[idx]
                image = batch['image'].unsqueeze(0).to(device)  # Add batch dimension
                
                # Generate prediction
                output = model(image)
                pred = torch.softmax(output, dim=1)
                pred = torch.argmax(pred, dim=1)
                pred = pred.cpu().numpy()[0]  # Remove batch dimension
                
                # Store prediction
                predictions[idx] = pred
        
        # Create NIFTI image and save
        nifti_img = nib.Nifti1Image(predictions, affine)
        output_path = output_dir / f'{case_id}_prediction.nii.gz'
        nib.save(nifti_img, output_path)
        print(f"Saved predicted mask to {output_path}")

if __name__ == '__main__':
    main() 