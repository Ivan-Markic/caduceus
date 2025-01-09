import click
import os
import SimpleITK as sitk
from pathlib import Path

@click.command()
@click.option('--data-dir', '-d', type=click.Path(exists=True), default='../kits19-challenge/kits19',
              help='Path to the directory containing imaging.nii.gz and segmentation.nii.gz')
@click.option('--output-dir', '-o', type=click.Path(), default=None,
              help='Path to save resized images. If not provided, will overwrite original files.')
@click.option('--case-filter', '-c', multiple=True, type=str, default=['case_00160'],
              help='Filter for case names to process. If not provided, will process all cases.')
def resize_image(data_dir, output_dir, case_filter):
    """
    Resize .nii.gz images to have dimensions [x, 512, 512].
    Will process both imaging.nii.gz and segmentation.nii.gz if present.
    """

    # Use input directory as output if not specified
    output_dir = output_dir or data_dir

    for case in os.listdir(data_dir):
        #
        if case_filter and case not in case_filter:
            print(f"Skipping case {case} because it is not in the case filter")
            continue

        print(f"Processing case {case}")
        
        # Load the original image
        image = sitk.ReadImage(f"{data_dir}/{case}/imaging.nii.gz")

        # Print the original image size and spacing
        original_size = image.GetSize()
        original_spacing = image.GetSpacing()
        print("Original size:", original_size)
        print("Original spacing:", original_spacing)


        # Define the desired output size
        # I want to keep number of slices the same
        new_size = [original_size[0], 512, 512]

        # Compute the new spacing to preserve physical dimensions
        new_spacing = [
            (original_size[i] * original_spacing[i]) / new_size[i] for i in range(3)
        ]

        print("New spacing:", new_spacing)

        # Compare spacings element by element
        if all(abs(new - orig) < 1e-10 for new, orig in zip(new_spacing, original_spacing)):
            print(f"Skipping case {case} because new spacing is same as original spacing")
            continue

        print("Resampling image")
        # Resample the image
        resampler = sitk.ResampleImageFilter()
        resampler.SetSize(new_size)
        resampler.SetOutputSpacing(new_spacing)
        resampler.SetOutputOrigin(image.GetOrigin())
        resampler.SetOutputDirection(image.GetDirection())
        resampler.SetInterpolator(sitk.sitkBSpline)  # Use BSpline interpolation for images

        resized_image = resampler.Execute(image)
        sitk.WriteImage(resized_image, f"{output_dir}/{case}/imaging.nii.gz")

        # Load and process the ROI mask if it exists
        mask_path = f"{data_dir}/{case}/segmentation.nii.gz"
        if Path(mask_path).exists():
            mask = sitk.ReadImage(mask_path)
            
            # Resample the mask using nearest-neighbor interpolation
            resampler.SetInterpolator(sitk.sitkNearestNeighbor)  # Use nearest-neighbor for masks
            resized_mask = resampler.Execute(mask)
            sitk.WriteImage(resized_mask, f"{output_dir}/{case}/segmentation.nii.gz")

            # Print the resized mask size
            print("Resized mask size:", resized_mask.GetSize())

        # Print the resized image size
        print("Resized image size:", resized_image.GetSize())

if __name__ == '__main__':
    resize_image()
