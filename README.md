# Perturbed Minds

## Conda environment and dependencies

To make running this code easier you can install the latest conda environment for this project stored in `perturbed-environment.yml`.

### Install environment from `perturbed-environment.yml`

If you want to install the latest conda environment run

`conda env create -f perturbed-environment.yml` 

### Install segmentation_models_pytorch newest version

PyPi version is not up-to-date with github version and lacks features

`python -m pip install git+https://github.com/qubvel/segmentation_models.pytorch`

### Update `perturbed-environment.yml`

If you add code that requires new packages, inside your perturbed-minds conda environment run

`conda env export > perturbed-environment.yml`

## Walk-through
Link to the repository structure we put down in miro: https://miro.com/app/board/o9J_lQdgyf8=/
