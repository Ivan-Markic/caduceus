#! /bin/bash

python scripts/resize_image.py -d ../kits19-challenge/kits19 -o preprocessed_data -c case_00160
python scripts/convert_data.py -d ../kits19-challenge/kits19 -o preprocessed_data
python scripts/predict.py -d ../kits19-challenge/kits19 --model-path checkpoints/unet/model_epoch_19_loss_0.5370.pth
python scripts/extract_pyradiomics_feature.py
