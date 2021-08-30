# From Lens to Logit - Addressing Camera Hardware-Drift Using Raw Sensor Data

*This repository hosts the code for the project ["From Lens to Logit: Addressing Camera Hardware-Drift Using Raw Sensor Data"](https://openreview.net/forum?id=DRAywM1BhU), submitted to the NeurIPS 2021 Datasets and Benchmarks Track.*

## A short introduction
To create an image, raw sensor data traverses complex image signal processing pipelines. These pipelines are used by cameras and scientific instruments to produce the images fed into machine learning systems. The processing pipelines vary by device, influencing the resulting image statistics and ultimately contributing to what is known as hardware-drift. However, this processing is rarely considered in machine learning modelling, because available benchmark data sets are generally not in raw format. Here we show that pairing qualified raw sensor data with an explicit, differentiable model of the image processing pipeline allows to tackle camera hardware-drift. Specifically, we demonstrate (1) the controlled synthesis of hardware-drift test cases, (2) modular hardware-drift forensics, as well as (3) image processing customization. We make available two data sets. The first, Raw-Microscopy, contains 940 raw bright-field microscopy images of human blood smear slides for leukocyte classification alongside 5,640 variations measured at six different intensities and twelve additional sets totalling 11,280 images of the raw sensor data processed through different pipelines. The second, Raw-Drone, contains 548 raw drone camera images for car segmentation, alongside 3,288 variations measured at six different intensities and also twelve additional sets totalling 6,576 images of the raw sensor data processed through different pipelines.

In order to address camera hardware-drift we require two ingredients: raw sensor data and an image processing model. This code repository contains the materials for the second ingredient, the image processing model, as well as scripts to load lada and run experiments. For a conceptual overview of the project we reocommend the [project site](https://aiaudit.org/lens2logit/) or the [full paper](https://openreview.net/forum?id=DRAywM1BhU).
## Data access
## Code
### Dependencies
#### Conda environment and dependencies
To make running this code easier you can install the latest conda environment for this project stored in `perturbed-environment.yml`.
##### Install environment from `perturbed-environment.yml`
If you want to install the latest conda environment run
`conda env create -f perturbed-environment.yml` 
##### Install segmentation_models_pytorch newest version
PyPi version is not up-to-date with github version and lacks features
`python -m pip install git+https://github.com/qubvel/segmentation_models.pytorch`
### Recreate experiments
### Review our experiments
### Use our trained models
