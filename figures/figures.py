import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
import argparse
#gif
import os
import pathlib
import shutil
import imageio
#plot
import matplotlib.pyplot as plt
import numpy as np

# -1. parse args
parser = argparse.ArgumentParser(description="results_analysis")
parser.add_argument("--tracking_uri", type=str,
                    default="http://deplo-mlflo-1ssxo94f973sj-890390d809901dbf.elb.eu-central-1.amazonaws.com", help='URI of the mlflow server on AWS')
parser.add_argument("--experiment_name", type=str, default=None,
                    help='Name of the experiment on the mlflow server, e.g. "processing_comparison"')
parser.add_argument("--run_name", type=str, default=None,
                    help='Name of the run on the mlflow server, e.g. "proc_nn"')
parser.add_argument("--representation", type=str, default=None,
                    choices=["processing", "gradients"], help='The representation form you want retrieve("processing" or "gradients")')
parser.add_argument("--step", type=str, default=None,
                    choices=["pre_debayer", "demosaic", "color_correct", "sharpening", "gaussian", "clipped", "gamma_correct", "rgb"], 
                    help='The processing step you want to track ("pre_debayer" or "rgb")') #TODO: include predictions and ground truths
parser.add_argument("--gif_name", type=str, default=None,
                    help='Name of the gif that will be saved. Note: .gif will be added later by script') #TODO: option to include filepath where result should be written
                    #TODO: option to write results to existing run on mlflow
parser.add_argument("--local_dir", type=str, default=None,
                    help='Name of the local dir to be created to store mlflow data')
parser.add_argument("--cleanup", type=bool, default=True,
                    help='Whether to delete the local dir again after the script was run')
parser.add_argument("--output", type=str, default=None,
                    choices=["gif", "train_vs_val_loss"],
                    help='Which output to generate') #TODO: make this cleaner, atm it is confusing because each figure may need different set of args and it is not clear how to manage that
                    #TODO: idea -> fix the types of args for each figure which define the figure type but parametrize those things that can reasonably vary
args = parser.parse_args()

# 0. mlflow basics
mlflow.set_tracking_uri(args.tracking_uri)

# 1. specify experiment_name, run_name, representation and step
#is done via parse_args

# 2. use get_experiment_by_name to get experiment object
experiment = mlflow.get_experiment_by_name(args.experiment_name)

# 3. extract experiment_id
#experiment.experiment_id

# 4. use search_runs with experiment_id and run_name for string search query
filter_string = "tags.mlflow.runName = '{}'".format(args.run_name) #create the filter string with using the runName tag to query mlflow
runs = mlflow.search_runs(experiment.experiment_id, filter_string=filter_string) #returns a pandas data frame where each row is a run (if several exist under that name)
client = MlflowClient() #TODO: look more into the options of client

if args.output == "gif": #TODO: outsource these options to functions which are then loaded and can be called
    # 5. extract run from list
    #TODO: parent run and cv option for analysis
    if args.local_dir:
        local_dir = args.local_dir+"/artifacts"
    else: #use the current working dir and make a subdir "artifacts" to store the data from mlflow
        local_dir = str(pathlib.Path().resolve())+"/artifacts"
    if not os.path.isdir('artifacts'):
        os.mkdir(local_dir) #create the local_dir if it does not exist, yet #TODO: more advanced catching of existing files etc
    dir = client.download_artifacts(runs["run_id"][0], "results", local_dir) #TODO: parametrize this number [0] so the right run is selected

    # 6. get filenames in chronological sequence and write them to gif
    dirs = [x[0] for x in os.walk(dir)]
    dirs = sorted(dirs, key=str.lower)[1:] #sort chronologically and remove parent dir from list

    with imageio.get_writer(args.gif_name+'.gif', mode='I') as writer: #https://imageio.readthedocs.io/en/stable/index.html#
        for epoch in dirs: #extract the right file from each epoch
            for _, _, files in os.walk(epoch): #
                for name in files:
                    if args.representation in name and args.step in name and "png" in name:
                        image = imageio.imread(epoch+"/"+name)
                        writer.append_data(image)

    # 7. cleanup the downloaded artifacts from client file system
    if args.cleanup:
        shutil.rmtree(local_dir) #delete the files downloaded from mlflow

elif args.output == "train_vs_val_loss":
    train_loss = client.get_metric_history(runs["run_id"][0], "train_loss") #returns a list of metric entities https://www.mlflow.org/docs/latest/_modules/mlflow/entities/metric.html
    val_loss = client.get_metric_history(runs["run_id"][0], "val_loss") #TODO: parametrize this number [0] so the right run is selected
    train_loss = sorted(train_loss, key=lambda m: m.step) #sort the metric objects in list according to step property
    val_loss = sorted(val_loss, key=lambda m: m.step)
    plt.figure()
    for m_train, m_val in zip(train_loss, val_loss):
        plt.scatter(m_train.value, m_val.value, alpha=1/(m_train.step+1), color='blue')
    plt.savefig("scatter.png") #TODO: parametrize filename
