import os
import csv
import SimpleITK as sitk
from radiomics import featureextractor

# Path to the folder containing all cases
data_folder = "../kits19-challenge/kits19"

# Initialize the feature extractor with default settings (no parameter file)
extractor = featureextractor.RadiomicsFeatureExtractor()

# Define output CSV file
output_csv = "from_scratch_unet_radiomics_features_from_predicted_masks.csv"

# Prepare the CSV file with headers
with open(output_csv, mode="w", newline="") as csv_file:
    writer = None  # Placeholder for CSV writer, set later after first extraction
    
    # Iterate through each case
    for case in sorted(os.listdir(data_folder)):
        print(case) 
        print(len(data_folder))
        case_path = os.path.join(data_folder, case)
        if not os.path.isdir(case_path):
            continue
        
        # Define paths for the image and segmentation
        image_path = os.path.join(case_path, "imaging.nii.gz")
        mask_path = os.path.join(case_path, f"{case}_prediction.nii.gz")

        # Check if both files exist for the case
        if not (os.path.exists(image_path) and os.path.exists(mask_path)):
            print(f"Skipping {case}: missing prediction mask or image")
            continue
        
        # Load the image and mask
        image = sitk.ReadImage(image_path)
        mask = sitk.ReadImage(mask_path)
        print(case) 
        # Extract features
        try:
            feature_vector = extractor.execute(image, mask)
            print(case) 
        except Exception as e:
            print(f"Error processing {case}: {e}")
            continue
        print(case) 
        # Prepare the row data for CSV
        row_data = {'Case': case}
        row_data.update(feature_vector)
        
        # Write to CSV with headers
        if writer is None:
            # First case: initialize CSV headers
            writer = csv.DictWriter(csv_file, fieldnames=row_data.keys())
            writer.writeheader()
        
        # Write row data for the current case
        writer.writerow(row_data)
        
        # Print status
        print(f"Processed {case}")

print(f"Feature extraction completed. Results saved to {output_csv}")
