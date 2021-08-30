# From Lens to Logit - Addressing Camera Hardware-Drift Using Raw Sensor Data

*This repository hosts the code for the project ["From Lens to Logit: Addressing Camera Hardware-Drift Using Raw Sensor Data"](https://openreview.net/forum?id=DRAywM1BhU), submitted to the NeurIPS 2021 Datasets and Benchmarks Track.*

In order to address camera hardware-drift we require two ingredients: raw sensor data and an image processing model. This code repository contains the materials for the second ingredient, the image processing model, as well as scripts to load lada and run experiments. For a conceptual overview of the project we reocommend the [project site](https://aiaudit.org/lens2logit/) or the [full paper](https://openreview.net/forum?id=DRAywM1BhU).

## A short introduction
<p align="center">
<img src="https://github.com/aiaudit-org/lens2logit/blob/master/readme/Slice%208.png">
</p>

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
    application_key_id = '003d6b042de536a0000000004'
    application_key = 'K003E5Cr+BAYlvSHfg2ynLtvS5aNq78'
    info = InMemoryAccountInfo()
    b2_api = B2Api(info)
    b2_api.authorize_account('production', application_key_id, application_key)
    bucket = b2_api.get_bucket_by_name(bucket_name)
    return bucket
```
We also maintain a copy of the entire dataset with a permanent identifier at Zenodo which you can find under 10.5281/zenodo.5235536.
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
## Virtual lab log
We maintain a collaborative virtual lab log at [this address](http://deplo-mlflo-1ssxo94f973sj-890390d809901dbf.elb.eu-central-1.amazonaws.com/#/). There you can browse experiment runs, analyze results through SQL queries and download trained processing and task models.
<p align="center">
<img src="https://github.com/aiaudit-org/lens2logit/blob/master/readme/mlflow%20(1).png">
</p>

### Review our experiments
Experiments are listed in the left column. You can select individual runs or compare metrics and parameters across different runs. For runs where we tracked images of intermediate processing steps and images of the gradients at these processing steps you can find at the bottom of a run page in the *results* folder for each epoch.
### Use our trained models
When selecting a run and a model was saved you can find the model files, state dictionary and instructions to load at the bottom of a run page under *models*. In the menu bar at the top of the virtual lab log you can also access models via the *Model Registry*. Our code is well integrated with the *mlflow* autologging and -loading package for PyTorch. So when using our code you can just specify the *model uri* as an argument and models will be fetched from the model registry automatically.
