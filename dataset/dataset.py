import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path

class CTDataset(Dataset):
    def __init__(self, data_dir, split='train', transform=None, case_filter=None):
        """
        Args:
            data_dir (str): Path to directory with images and masks
            split (str): 'train', 'valid', or 'test'
            transform (callable, optional): Optional transform to be applied
            case_filter (str, optional): Filter for specific case_id
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        
        # Define case ranges based on split
        case_ranges = {
            'train': range(63, 210),
            'valid': range(0, 63),
            'test': range(210, 300),
            'eval': range(0, 210),
            'predict': range(0, 300)
        }

        # Dictionary to store paths organized by case
        self.cases = {}
        
        for case_id in case_ranges[split]:
            case_name = f'case_{case_id:05d}'
            case_dir = self.data_dir / case_name
            
            # Skip if case directory doesn't exist
            if not case_dir.exists():
                print(f"Warning: Case directory not found: {case_dir}")
                continue
                
            # Apply case filter if provided
            if case_filter and case_name != case_filter:
                continue
            
            # Get all slices for this case
            image_paths = sorted(list((case_dir / 'images').glob('slice_*.npy')))
            if not image_paths:
                print(f"Warning: No image slices found in {case_dir / 'images'}")
                continue

            # Get affine matrix from case_dir / affine.npy
            affine = np.load(case_dir / 'affine.npy')
            
            # Get mask paths if needed
            mask_paths = []
            if split != 'test' and split != 'predict':
                mask_paths = sorted(list((case_dir / 'masks').glob('slice_*.npy')))
                if len(image_paths) != len(mask_paths):
                    raise ValueError(f"Number of images ({len(image_paths)}) and masks ({len(mask_paths)}) don't match for {case_name}")
            
            # Store paths for this case
            self.cases[case_name] = {
                'image_paths': image_paths,
                'mask_paths': mask_paths,
                'affine': affine
            }
        
        if len(self.cases) == 0:
            raise ValueError(f"No data found for split '{split}' in {data_dir}")
        
        # Store case_ids for easy access
        self.case_ids = sorted(list(self.cases.keys()))

    def get_case_slices(self, case_id):
        """Get number of slices for a specific case."""
        return len(self.cases[case_id]['image_paths'])

    def get_case_data(self, case_id, slice_idx):
        """Get specific slice from a case."""
        case = self.cases[case_id]
        
        # Load image
        image = np.array(np.load(case['image_paths'][slice_idx]))
        image = np.expand_dims(image, axis=0)
        
        # Create return dictionary
        data = {
            'image': torch.from_numpy(image).float(),
            'path': str(case['image_paths'][slice_idx]),
            'case_id': case_id,
            'affine': case['affine']
        }
        
        # Add mask if available
        if case['mask_paths']:
            mask = np.array(np.load(case['mask_paths'][slice_idx]))
            mask = np.expand_dims(mask, axis=0)
            
            if self.transform is not None:
                transformed = self.transform(image=image, mask=mask)
                image = transformed['image']
                mask = transformed['mask']
            
            data['mask'] = torch.from_numpy(mask).long()
        elif self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed['image']
            
        data['image'] = torch.from_numpy(image).float()
        return data

    def __len__(self):
        """Total number of slices across all cases."""
        return len(self.cases)

    def __getitem__(self, idx):
        """Get a slice by global index."""
        # Find which case this index belongs to
        for case_id in self.case_ids:
            case_len = len(self.cases[case_id]['image_paths'])
            if idx < case_len:
                return self.get_case_data(case_id, idx)
            idx -= case_len
        raise IndexError("Index out of range")

    def get_affine(self, case_id):
        """Get affine matrix for a specific case.
        
        Args:
            case_id (str): Case ID (e.g., 'case_00000')
            
        Returns:
            numpy.ndarray: 4x4 affine matrix from the original nifti file
        """
        return self.cases[case_id]['affine']
        