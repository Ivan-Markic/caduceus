import numpy as np
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform

class CTWindow(ImageOnlyTransform):
    """Apply CT windowing to the image."""
    
    def __init__(self, window_center=40, window_width=80, always_apply=True, p=1.0):
        super().__init__(always_apply=always_apply, p=p)
        self.window_center = window_center
        self.window_width = window_width
    
    def apply(self, image, **params):
        min_value = self.window_center - self.window_width // 2
        max_value = self.window_center + self.window_width // 2
        return np.clip(image, min_value, max_value)

class MinMaxNormalize(ImageOnlyTransform):
    """Normalize image to [0, 1] range."""
    
    def __init__(self, always_apply=True, p=1.0):
        super().__init__(always_apply=always_apply, p=p)
    
    def apply(self, image, **params):
        min_val = image.min()
        max_val = image.max()
        if max_val - min_val != 0:
            return (image - min_val) / (max_val - min_val)
        return image

def get_training_transforms():
    return A.Compose([
        CTWindow(window_center=40, window_width=80),
        MinMaxNormalize(),
        A.ShiftScaleRotate(
            shift_limit=0.1,
            scale_limit=0.1,
            rotate_limit=15,
            border_mode=0,
            p=0.5
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.GaussNoise(var_limit=(0, 0.01), p=0.2),
        A.ElasticTransform(
            alpha=120,
            sigma=120 * 0.05,
            alpha_affine=120 * 0.03,
            p=0.3
        ),
    ])

def get_validation_transforms():
    return A.Compose([
        CTWindow(window_center=40, window_width=80),
        MinMaxNormalize(),
    ])
