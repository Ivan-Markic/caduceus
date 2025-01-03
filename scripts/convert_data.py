import click
import nibabel as nib
import numpy as np
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm

def normalize_image(image):
    min_val = image.min()
    max_val = image.max()
    if max_val - min_val != 0:
        return (image - min_val) / (max_val - min_val)
    return image

def convert_case(args):
    """Convert a single case"""
    case_dir, output_dir = args
    
    case_id = case_dir.name
    case_num = int(case_id.split('_')[1])
    
    # Create case-specific directories
    case_output_dir = output_dir / case_id
    
    # Check if case is already processed
    if case_output_dir.exists():
        # Check if at least one slice exists
        if list((case_output_dir / 'images').glob('slice_*.npy')):
            print(f"Skipping {case_id}: already processed")
            return
    
    # Create directories if they don't exist
    case_output_dir.mkdir(exist_ok=True)
    (case_output_dir / 'images').mkdir(exist_ok=True)
    (case_output_dir / 'masks').mkdir(exist_ok=True)
    
    # Load image
    image_path = case_dir / 'imaging.nii.gz'
    if not image_path.exists():
        print(f"Warning: No imaging file found for {case_id}")
        return
        
    image_nii = nib.load(str(image_path))
    image_data = image_nii.get_fdata()
    
    # Save affine matrix
    np.save(
        case_output_dir / 'affine.npy',
        image_nii.affine
    )
    
    # Process each slice
    for slice_idx in range(image_data.shape[0]):
        image_slice = image_data[slice_idx, :, :]
        
        # Save raw image slice
        np.save(
            case_output_dir / 'images' / f'slice_{slice_idx:03d}.npy',
            image_slice.astype(np.float32)
        )
        
        # Save mask slice if not a test case
        if case_num < 210:  # Not a test case
            mask_path = case_dir / 'segmentation.nii.gz'
            if mask_path.exists():
                mask_nii = nib.load(str(mask_path))
                mask_data = mask_nii.get_fdata()
                mask_slice = mask_data[slice_idx, :, :]
                np.save(
                    case_output_dir / 'masks' / f'slice_{slice_idx:03d}.npy',
                    mask_slice.astype(np.uint8)
                )

@click.command()
@click.option('--data-dir', type=click.Path(exists=True), required=True,
              help='Path to KiTS19 data directory')
@click.option('--output-dir', type=click.Path(), required=True,
              help='Path to save converted .npy files')
@click.option('--num-workers', default=-1, help='Number of worker processes (-1 for all cores)')
@click.option('--num-cases', default=0, help='Number of cases to process (0 for all)')
def convert_dataset(data_dir, output_dir, num_workers, num_cases):
    """Convert KiTS19 .nii.gz files to preprocessed .npy slices using parallel processing."""
    
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all case directories
    cases = sorted([d for d in data_dir.iterdir() if d.is_dir()])

    # Limit cases if specified
    if num_cases > 0:
        cases = cases[:num_cases]

    # Set number of workers
    if num_workers <= 0:
        num_workers = mp.cpu_count()

    # Prepare arguments for parallel processing
    args = [(case, output_dir) for case in cases]

    # Process cases in parallel with progress bar
    with mp.Pool(num_workers) as pool:
        list(tqdm(pool.imap(convert_case, args), total=len(args), desc="Converting cases"))

if __name__ == '__main__':
    convert_dataset()