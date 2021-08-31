#!/bin/bash

datasets='Microscopy Drone'
augmentations='weak strong none'

for augment in $augmentations
    do
    for data in $datasets
        do
        
        python show_results.py \
        --dataset $data \
        --augmentation $augment \

    done
done