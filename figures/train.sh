#!/bin/bash

# # Parametrized Training
# 100 epochs, frozen_processor: http://deplo-mlflo-1ssxo94f973sj-890390d809901dbf.elb.eu-central-1.amazonaws.com/#/experiments/49/runs/2803f44514e34a0f87d591520706e876
# model_uri="s3://mlflow-artifacts-601883093460/49/2803f44514e34a0f87d591520706e876/artifacts/model"

# used for training current model to 100% train and 80% val accuracy
# python train.py \
# --experiment_name parametrized \
# --classifier_uri "${model_uri}" \
# --run_name par_full_kurt \
# --dataset Microscopy \
# --lr 1e-5 \
# --epochs 50 \
# --freeze_classifier \

# --freeze_processor \

# # Adversarial Training

# python train.py \
# --experiment_name adversarial \
# --run_name adv_frozen_processor \
# --classifier_uri "${model_uri}" \
# --dataset Microscopy \
# --adv_training \
# --lr 1e-3 \
# --epochs 7 \
# --freeze_classifier \
# --track_processing \
# --track_every_epoch \
# --log_model=False \
# --adv_aux_weight=0.1 \
# --adv_aux_loss "l2" \

# --adv_aux_weight=2e-5 \
# --adv_aux_weight=2e-5 \
# --adv_aux_weight=1.9e-5 \

# Cross pipeline training (Segmentation/Classification)

# Static Pipeline Script        

# datasets="Microscopy Drone DroneSegmentation"
datasets="DroneSegmentation"
augmentations="weak strong none"

demosaicings="bilinear malvar2004 menon2007"
sharpenings="sharpening_filter unsharp_masking"
denoisings="median_denoising gaussian_denoising"

for augment in $augmentations
    do
    for data in $datasets
        do
        for demosaicing in $demosaicings 
            do
            for sharpening in $sharpenings
                do
                for denoising in $denoisings
                    do

                    python train.py \
                    --experiment_name ABtesting \
                    --run_name "$data"_"$demosaicing"_"$sharpening"_"$denoising"_"$augment" \
                    --dataset "$data" \
                    --batch_size 4 \
                    --lr 1e-5 \
                    --epochs 100 \
                    --sp_debayer "$demosaicing" \
                    --sp_sharpening "$sharpening" \
                    --sp_denoising "$denoising" \
                    --processing_mode "static" \
                    --augmentation "$augment" \
                    --n_split 5 \
                    
                done
            done
        done
    done
done