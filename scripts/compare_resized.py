import click
import nibabel as nib
import numpy as np
from pathlib import Path
from skimage.transform import resize
from tqdm import tqdm

def resize_volume(volume, target_shape=(512, 512)):
    """Resize each slice in the volume."""
    resized_slices = []
    for i in range(volume.shape[0]):
        slice_2d = volume[i, :, :]
        resized_slice = resize(slice_2d, target_shape, mode='constant', anti_aliasing=True)
        resized_slices.append(resized_slice)
    return np.stack(resized_slices)

@click.command()
@click.option('--case-dir', type=click.Path(exists=True), required=True,
              help='Path to the specific case directory containing imaging.nii.gz')
@click.option('--output-dir', type=click.Path(), required=True,
              help='Path to save processed files')
def main(case_dir, output_dir):
    case_dir = Path(case_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load original image
    image_path = case_dir / 'imaging.nii.gz'
    mask_path = case_dir / 'segmentation.nii.gz'
    
    if not image_path.exists():
        print(f"No imaging file found at {image_path}")
        return
        
    print("Loading original image...")
    image_nii = nib.load(str(image_path))
    image_data = image_nii.get_fdata()
    original_shape = image_data.shape
    print(f"Original shape: {original_shape}")
    
    # Resize volume
    print("Resizing volume...")
    resized_image = resize_volume(image_data, target_shape=(512, 512))
    print(f"Resized shape: {resized_image.shape}")
    
    # Save as .npy files
    print("Saving .npy files...")
    npy_dir = output_dir / 'npy_files'
    npy_dir.mkdir(exist_ok=True)
    
    for i in tqdm(range(resized_image.shape[0])):
        np.save(
            npy_dir / f'slice_{i:03d}.npy',
            resized_image[i].astype(np.float32)
        )
    
    # Save resized volume as .nii.gz
    print("Saving resized .nii.gz...")
    resized_nii = nib.Nifti1Image(resized_image, image_nii.affine)
    nib.save(resized_nii, str(output_dir / 'resized_imaging.nii.gz'))
    
    # Process mask if it exists
    if mask_path.exists():
        print("Processing mask...")
        mask_nii = nib.load(str(mask_path))
        mask_data = mask_nii.get_fdata()
        
        # Resize mask
        resized_mask = resize_volume(mask_data, target_shape=(512, 512))
        
        # Round mask values to nearest integer after resize
        resized_mask = np.round(resized_mask).astype(np.uint8)
        
        # Save resized mask
        resized_mask_nii = nib.Nifti1Image(resized_mask, mask_nii.affine)
        nib.save(resized_mask_nii, str(output_dir / 'resized_segmentation.nii.gz'))
    
    print("\nProcessing complete!")
    print(f"Original files: {case_dir}")
    print(f"Processed files: {output_dir}")
    print("\nYou can now compare these in 3D Slicer:")
    print(f"1. Original: {image_path}") 
    print(f"2. Resized: {output_dir / 'resized_imaging.nii.gz'}")

if __name__ == '__main__':
    main() 