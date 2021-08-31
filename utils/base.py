"""
Utilities for other scripts
"""

import os
import shutil

import random

import torch
import mlflow
from mlflow.tracking import MlflowClient
import numpy as np

from IPython.display import display, Markdown

from b2sdk.v1 import *

import argparse


class SmartFormatter(argparse.HelpFormatter):
        
    def _split_lines(self, text, width):
        if text.startswith('R|'):
            return text[2:].splitlines()  
        # this is the RawTextHelpFormatter._split_lines
        return argparse.HelpFormatter._split_lines(self, text, width)


def str2bool(string):
    return string == 'True'


def np2torch(nparray):
    """Convert numpy array to torch tensor
       For array with more than 3 channels, it is better to use an input array in the format BxHxWxC

       Args:
           numpy array (ndarray) BxHxWxC
       Returns:
           torch tensor (tensor) BxCxHxW"""

    tensor = torch.Tensor(nparray)

    if tensor.ndim == 2:
        return tensor
    if tensor.ndim == 3:
        height, width, channels = tensor.shape
        if channels <= 3:  # Single image with more channels (HxWxC)
            return tensor.permute(2, 0, 1)

    if tensor.ndim == 4:  # More images with more channels (BxHxWxC)
        return tensor.permute(0, 3, 1, 2)

    return tensor


def torch2np(torchtensor):
    """Convert torch tensor to numpy array 
       For tensor with more than 3 channels or batch, it is better to use an input tensor in the format BxCxHxW

       Args:
           torch tensor (tensor) BxCxHxW
       Returns:
           numpy array (ndarray) BxHxWxC"""

    ndarray = torchtensor.detach().cpu().numpy().astype(np.float32)

    if ndarray.ndim == 3:  # Single image with more channels (CxHxW)
        channels, height, width = ndarray.shape
        if channels <= 3:
            return ndarray.transpose(1, 2, 0)

    if ndarray.ndim == 4:  # More images with more channels (BxCxHxW)
        return ndarray.transpose(0, 2, 3, 1)

    return ndarray


def set_random_seed(seed):
    np.random.seed(seed)  # cpu vars
    torch.manual_seed(seed)  # cpu  vars
    random.seed(seed)  # Python
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # gpu vars
        torch.backends.cudnn.deterministic = True  # needed
        torch.backends.cudnn.benchmark = False


def normalize(img):
    """Normalize images

       Args:
            imgs (ndarray): image to normalize --> size: (Height,Width,Channels)
       Returns:
            normalized (ndarray): normalized image
            mu (ndarray): mean
            sigma (ndarray): standard deviation
    """

    img = img.astype(float)

    if len(img.shape) == 2:
        img = img[:, :, np.newaxis]

    height, width, channels = img.shape

    mu, sigma = np.empty(channels), np.empty(channels)

    for ch in range(channels):
        temp_mu = img[:, :, ch].mean()
        temp_sigma = img[:, :, ch].std()

        img[:, :, ch] = (img[:, :, ch] - temp_mu) / (temp_sigma + 1e-4)

        mu[ch] = temp_mu
        sigma[ch] = temp_sigma

    return img, mu, sigma


def b2_list_files(folder=''):
    bucket = get_b2_bucket()
    for file_info, _ in bucket.ls(folder, show_versions=False):
        print(file_info.file_name)


def get_b2_bucket():
    bucket_name = 'perturbed-minds'
    application_key_id = '003d6b042de536a0000000004'
    application_key = 'K003E5Cr+BAYlvSHfg2ynLtvS5aNq78'
    info = InMemoryAccountInfo()
    b2_api = B2Api(info)
    b2_api.authorize_account('production', application_key_id, application_key)
    bucket = b2_api.get_bucket_by_name(bucket_name)
    return bucket


def b2_download_folder(b2_dir, local_dir, force_download=False, mirror_folder=True):
    """Downloads a folder from the b2 bucket and optionally cleans
    up files that are no longer on the server

    Args:
        b2_dir (str): path to folder on the b2 server
        local_dir (str): path to folder on the local machine
        force_download (bool, optional): force the download, if set to `False`, 
            files with matching names on the local machine will be skipped
        mirror_folder (bool, optional): if set to `True`, files that are found in
            the local directory, but are not on the server will be deleted
    """
    bucket = get_b2_bucket()

    if not os.path.exists(local_dir):
        os.makedirs(local_dir)
    elif not force_download:
        return

    download_files = [file_info.file_name.split(b2_dir + '/')[-1]
                      for file_info, _ in bucket.ls(b2_dir, show_versions=False)]

    for file_name in download_files:
        if file_name.endswith('/.bzEmpty'):  # subdirectory, download recursively
            subdir = file_name.replace('/.bzEmpty', '')
            if len(subdir) > 0:
                b2_subdir = os.path.join(b2_dir, subdir)
                local_subdir = os.path.join(local_dir, subdir)
                if b2_subdir != b2_dir:
                    b2_download_folder(b2_subdir, local_subdir, force_download=force_download,
                                       mirror_folder=mirror_folder)
        else:   # file
            b2_file = os.path.join(b2_dir, file_name)
            local_file = os.path.join(local_dir, file_name)
            if not os.path.exists(local_file) or force_download:
                print(f"downloading b2://{b2_file} -> {local_file}")
                bucket.download_file_by_name(b2_file, DownloadDestLocalFile(local_file))

    if mirror_folder:   # remove all files that are not on the b2 server anymore
        for i, file in enumerate(download_files):
            if file.endswith('/.bzEmpty'):  # subdirectory, download recursively
                download_files[i] = file.replace('/.bzEmpty', '')
        for file_name in os.listdir(local_dir):
            if file_name not in download_files:
                local_file = os.path.join(local_dir, file_name)
                print(f"deleting {local_file}")
                if os.path.isdir(local_file):
                    shutil.rmtree(local_file)
                else:
                    os.remove(local_file)


def get_name(obj):
    return obj.__name__ if hasattr(obj, '__name__') else type(obj).__name__


def get_mlflow_model_by_name(experiment_name, run_name, 
                    tracking_uri = "http://deplo-mlflo-1ssxo94f973sj-890390d809901dbf.elb.eu-central-1.amazonaws.com",
                    download_model = True):
    
    # 0. mlflow basics
    mlflow.set_tracking_uri(tracking_uri)
    os.environ["AWS_ACCESS_KEY_ID"] = "#TODO: add your AWS access key if you want to write your results to our collaborative lab server"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "#TODO: add your AWS seceret access key if you want to write your results to our collaborative lab server"

    # # 1. use get_experiment_by_name to get experiment objec
    experiment = mlflow.get_experiment_by_name(experiment_name)

    # # 2. use search_runs with experiment_id for string search query
    if os.path.isfile('cache/runs_names.pkl'):
        runs = pd.read_pickle('cache/runs_names.pkl')
        if runs['tags.mlflow.runName'][runs['tags.mlflow.runName'] == run_name].empty:
            runs = fetch_runs_list_mlflow(experiment) #returns a pandas data frame where each row is a run (if several exist under that name) 
    else: 
        runs = fetch_runs_list_mlflow(experiment) #returns a pandas data frame where each row is a run (if several exist under that name) 

    # 3. get the selected run between all runs inside the selected experiment
    run = runs.loc[runs['tags.mlflow.runName'] == run_name]

    # 4. check if there is only a run with that name
    assert len(run) == 1, "More runs with this name"
    index_run = run.index[0]
    artifact_uri = run.loc[index_run, 'artifact_uri']

    # 5. load state_dict of your run
    state_dict = mlflow.pytorch.load_state_dict(artifact_uri)

    # 6. load model of your run
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    # model = mlflow.pytorch.load_model(os.path.join(
    #         artifact_uri, "model"), map_location=torch.device(DEVICE))
    model = fetch_from_mlflow(os.path.join(
                artifact_uri, "model"), use_cache=True, download_model=download_model)

    return state_dict, model

def data_loader_mean_and_std(data_loader, transform=None):
    means = []
    stds = []
    for x, y in data_loader:
        if transform is not None:
            x = transform(x)
        means.append(x.mean(dim=(0, 2, 3)).unsqueeze(0))
        stds.append(x.std(dim=(0, 2, 3)).unsqueeze(0))
    return torch.cat(means).mean(dim=0), torch.cat(stds).mean(dim=0)

def fetch_runs_list_mlflow(experiment):        
    runs = mlflow.search_runs(experiment.experiment_id)
    runs.to_pickle('cache/runs_names.pkl')  # where to save it, usually as a .pkl
    return runs

def fetch_from_mlflow(uri, use_cache=True, download_model=True):
    cache_loc = os.path.join('cache', uri.split('//')[1]) + '.pt'
    if use_cache and os.path.exists(cache_loc):
        print(f'loading cached model from {cache_loc} ...')
        return torch.load(cache_loc)
    else:
        print(f'fetching model from {uri} ...')
        model = mlflow.pytorch.load_model(uri)
        os.makedirs(os.path.dirname(cache_loc), exist_ok=True)
        if download_model:
            torch.save(model, cache_loc, pickle_module=mlflow.pytorch.pickle_module)
        return model


def display_mlflow_run_info(run):
    uri = mlflow.get_tracking_uri()
    experiment_id = run.info.experiment_id
    experiment_name = mlflow.get_experiment(experiment_id).name
    run_id = run.info.run_id
    run_name = run.data.tags['mlflow.runName']
    experiment_url = f'{uri}/#/experiments/{experiment_id}'
    run_url = f'{experiment_url}/runs/{run_id}'

    print(f'view results at {run_url}')
    display(Markdown(
        f"[<a href='{experiment_url}'>experiment {experiment_id} '{experiment_name}'</a>]"
        f" > "
        f"[<a href='{run_url}'>run '{run_name}' {run_id}</a>]"
    ))
    print('')


def get_train_test_indices_drone(df, frac, seed=None):
    """ Split indices of a DataFrame with binary and balanced labels into balanced subindices

   Args:
        df (pd.DataFrame): {0,1}-labeled data
        frac (float): fraction of indicies in first subset
        random_seed (int): random seed used as random state in np.random and as argument for random.seed()
   Returns:
       train_indices (torch.tensor): balanced subset of indices corresponding to rows in the DataFrame
       test_indices (torch.tensor): balanced subset of indices corresponding to rows in the DataFrame
    """

    split_idx = int(len(df) * frac / 2)
    df_with = df[df['label'] == 1]
    df_without = df[df['label'] == 0]

    np.random.seed(seed)
    df_with_train = df_with.sample(n=split_idx, random_state=seed)
    df_with_test = df_with.drop(df_with_train.index)

    df_without_train = df_without.sample(n=split_idx, random_state=seed)
    df_without_test = df_without.drop(df_without_train.index)

    train_indices = list(df_without_train.index) + list(df_with_train.index)
    test_indices = list(df_without_test.index) + list(df_with_test.index)

    """"
    print('fraction of 1-label in train set: {}'.format(len(df_with_train)/(len(df_with_train) + len(df_without_train))))
    print('fraction of 1-label in test set: {}'.format(len(df_with_test)/(len(df_with_test) + len(df_with_test))))
    """

    return train_indices, test_indices


def smp_get_loss(loss):
    if loss == "Dice":
        return smp.losses.DiceLoss(mode='binary', from_logits=True)
    if loss == "BCE":
        return nn.BCELoss()
    elif loss == "BCEWithLogits":
        return smp.losses.BCEWithLogitsLoss()
    elif loss == "DicyBCE":
        from pytorch_toolbelt import losses as ptbl
        return ptbl.JointLoss(ptbl.DiceLoss(mode='binary', from_logits=False),
                              nn.BCELoss(),
                              first_weight=args.dice_weight,
                              second_weight=args.bce_weight)
