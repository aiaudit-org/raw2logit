[![MIT License](https://img.shields.io/apm/l/atomic-design-ui.svg?)](https://github.com/tterb/atomic-design-ui/blob/master/LICENSEs) [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5235536.svg)](https://doi.org/10.5281/zenodo.5235536)

# From Lens to Logit - Addressing Camera Hardware-Drift Using Raw Sensor Data

*This repository hosts the code for the project ["From Lens to Logit: Addressing Camera Hardware-Drift Using Raw Sensor Data"](https://openreview.net/forum?id=DRAywM1BhU), submitted to the NeurIPS 2021 Datasets and Benchmarks Track.*

In order to address camera hardware-drift we require two ingredients: raw sensor data and an image processing model. This code repository contains the materials for the second ingredient, the image processing model, as well as scripts to load lada and run experiments. For a conceptual overview of the project we reocommend the [project site](https://aiaudit.org/lens2logit/) or the [full paper](https://openreview.net/forum?id=DRAywM1BhU).

## A short introduction
![L2L Overview](https://user-images.githubusercontent.com/38631399/131536063-585cf9b0-e76e-4e41-a05e-2fcf4902f539.png)


To create an image, raw sensor data traverses complex image signal processing pipelines. These pipelines are used by cameras and scientific instruments to produce the images fed into machine learning systems. The processing pipelines vary by device, influencing the resulting image statistics and ultimately contributing to what is known as hardware-drift. However, this processing is rarely considered in machine learning modelling, because available benchmark data sets are generally not in raw format. Here we show that pairing qualified raw sensor data with an explicit, differentiable model of the image processing pipeline allows to tackle camera hardware-drift. 

Specifically, we demonstrate 
1. the **controlled synthesis of hardware-drift test cases**
2. modular **hardware-drift forensics**, as well as 
3. **image processing customization**. 

We make available two data sets. 
1. **Raw-Microscopy**, contains 
   * **940 raw bright-field microscopy images** of human blood smear slides for leukocyte classification alongside 
   * **5,640 variations measured at six different intensities** and twelve additional sets totalling 
   * **11,280 images of the raw sensor data processed through different pipelines**.
3. **Raw-Drone**, contains 
   * **548 raw drone camera images** for car segmentation, alongside 
   * **3,288 variations measured at six different intensities** and also twelve additional sets totalling 
   * **6,576 images of the raw sensor data processed through different pipelines**.
## Data access
If you use our code you can use the convenient cloud storage integration. Data will be loaded automatically from a cloud storage bucket and stored to your working machine. You can find the code snippet doing that [here](https://github.com/aiaudit-org/lens2logit/blob/f8a165a0c094456f68086167f0bef14c3b311a4e/utils/base.py#L130)

```python
def get_b2_bucket():
    bucket_name = 'perturbed-minds'
<<<<<<< HEAD
    application_key_id = '003d6b042de536a0000000004'
    application_key = 'K003E5Cr+BAYlvSHfg2ynLtvS5aNq78'
=======
    application_key_id = '003d6b042de536a0000000008'
    application_key = 'K003HMNxnoa91Dy9c0V8JVCKNUnwR9U'
>>>>>>> ea1d33b387781225b4149b4b1b3b04f34dc42268
    info = InMemoryAccountInfo()
    b2_api = B2Api(info)
    b2_api.authorize_account('production', application_key_id, application_key)
    bucket = b2_api.get_bucket_by_name(bucket_name)
    return bucket
```
We also maintain a copy of the entire dataset with a permanent identifier at Zenodo which you can find here [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5235536.svg)](https://doi.org/10.5281/zenodo.5235536).
## Code
### Dependencies
#### Conda environment and dependencies
To run this code out-of-the-box you can install the latest project conda environment stored in `environment.yml`
```console
$ conda env create -f environment.yml
```
#### segmentation_models_pytorch newest version
We noticed that PyPi package for `segmentation_models_pytorch` is sometimes behind the project's github repository. If you encounter `smp` related problems we reccomend installing directly from the `smp`  reposiroty via
```console
$ python -m pip install git+https://github.com/qubvel/segmentation_models.pytorch
```
### Recreate experiments
The central file for using the **Lens2Logit** framework for experiments as in the paper is `train.py` which provides a rich set of arguments to experiment with raw image data, different image processing models and task models for regression or classification. Below we provide three example prompts for the type of experiments reported in the [paper](https://openreview.net/forum?id=DRAywM1BhU)
#### Controlled synthesis of hardware-drift test cases
```console
$ python train.py \
--experiment_name YOUR-EXPERIMENT-NAME \
--run_name YOUR-RUN-NAME \
--dataset Microscopy \
--lr 1e-5 \
--n_splits 5 \
--epochs 5 \
--classifier_pretrained \
--processing_mode static \
--augmentation weak \
--log_model True \
--iso 0.01 \
--freeze_processor \
--processor_uri "$processor_uri" \
--track_processing \
--track_every_epoch \
--track_predictions \
--track_processing_gradients \
--track_save_tensors \
```
#### Modular hardware-drift forensics
```console
$ python train.py \
--experiment_name YOUR-EXPERIMENT-NAME \
--run_name YOUR-RUN-NAME \
--dataset Microscopy \
--adv_training
--lr 1e-5 \
--n_splits 5 \
--epochs 5 \
--classifier_pretrained \
--processing_mode parametrized \
--augmentation weak \
--log_model True \
--iso 0.01 \
--track_processing \
--track_every_epoch \
--track_predictions \
--track_processing_gradients \
--track_save_tensors \
```
#### Image processing customization
```console
$ python train.py \
--experiment_name YOUR-EXPERIMENT-NAME \
--run_name YOUR-RUN-NAME \
--dataset Microscopy \
--lr 1e-5 \
--n_splits 5 \
--epochs 5 \
--classifier_pretrained \
--processing_mode parametrized \
--augmentation weak \
--log_model True \
--iso 0.01 \
--track_processing \
--track_every_epoch \
--track_predictions \
--track_processing_gradients \
--track_save_tensors \
```
## Virtual lab log
We maintain a collaborative virtual lab log at [this address](http://deplo-mlflo-1ssxo94f973sj-890390d809901dbf.elb.eu-central-1.amazonaws.com/#/). There you can browse experiment runs, analyze results through SQL queries and download trained processing and task models.
![mlflow](https://user-images.githubusercontent.com/38631399/131536233-f6b6e0ae-35f2-4ee0-a5e2-d04f8efb8d73.png)


### Review our experiments
Experiments are listed in the left column. You can select individual runs or compare metrics and parameters across different runs. For runs where we tracked images of intermediate processing steps and images of the gradients at these processing steps you can find at the bottom of a run page in the *results* folder for each epoch.
### Use our trained models
When selecting a run and a model was saved you can find the model files, state dictionary and instructions to load at the bottom of a run page under *models*. In the menu bar at the top of the virtual lab log you can also access models via the *Model Registry*. Our code is well integrated with the *mlflow* autologging and -loading package for PyTorch. So when using our code you can just specify the *model uri* as an argument and models will be fetched from the model registry automatically.
