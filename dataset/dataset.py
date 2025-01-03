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
            'test': range(210, 300)
        }
        
        # Get all slice paths for the given split
        self.image_paths = []
        self.mask_paths = []
        
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
            
            self.image_paths.extend(image_paths)
            
            if split != 'test':
                mask_paths = sorted(list((case_dir / 'masks').glob('slice_*.npy')))
                if len(image_paths) != len(mask_paths):
                    raise ValueError(f"Number of images ({len(image_paths)}) and masks ({len(mask_paths)}) don't match for {case_name}")
                self.mask_paths.extend(mask_paths)
        
        if len(self.image_paths) == 0:
            raise ValueError(f"No data found for split '{split}' in {data_dir}")

    def get_affine(self, case_id):
        """Get affine matrix for a given case."""
        affine_path = self.data_dir / case_id / 'affine.npy'
        return np.load(affine_path)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image and create a new array
        image = np.array(np.load(self.image_paths[idx]))
        
        # Add channel dimension if needed
        image = np.array(np.expand_dims(image, axis=0))
        
        # Get case_id and affine
        case_id = self.image_paths[idx].parent.parent.name
        affine = self.get_affine(case_id)
        
        # For test set, we don't load masks
        if self.split == 'test':
            if self.transform is not None:
                transformed = self.transform(image=image)
                image = np.array(transformed['image'])
            
            return {
                'image': torch.from_numpy(image).float(),
                'path': str(self.image_paths[idx]),
                'case_id': case_id,
                'affine': affine
            }
        
        # For train/valid, load both image and mask
        mask = np.array(np.load(self.mask_paths[idx]))
        mask = np.array(np.expand_dims(mask, axis=0))
        
        # Apply transforms if any
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = np.array(transformed['image'])
            mask = np.array(transformed['mask'])
        
        return {
            'image': torch.from_numpy(image).float(),
            'mask': torch.from_numpy(mask).long(),
            'path': str(self.image_paths[idx]),
            'case_id': case_id,
            'affine': affine
        }