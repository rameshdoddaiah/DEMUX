# To add a new markdown cell, type '# %% [markdown]'
# %%
from munch import Munch
# from IPython import get_ipython
import torch
import sys
import os
import wandb
import ssl
from nte.train_black_box_model import train_black_box_model
# All 9 datasets
from nte.data.synth.blipv3 import BlipV3Dataset
import shortuuid
import argparse
from nte.data.real.univariate.binary.wafer import WaferDataset
from nte.data.real.univariate.binary.earthquakes import EarthquakesDataset

# def send_plt_to_wandb(plt, title):
#     buf = io.BytesIO()
#     plt.savefig(buf, format='png')
#     buf.seek(0)
#     return wandb.Image(Image.open(buf), caption=title)


# get_ipython().run_line_magic('load_ext', 'autoreload')
# get_ipython().run_line_magic('autoreload', '2')

# Instantiate the parser

ssl._create_default_https_context = ssl._create_unverified_context
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor

ENABLE_WANDB = True
# WANDB_DRY_RUN = True
WANDB_DRY_RUN = False

BASE_SAVE_DIR = 'results/0109/'
PROJECT_NAME = "timeseries-saliency-blackbox"
# PROJECT_NAME = "time_series-cm-mse-lime-saliency"

if WANDB_DRY_RUN:
    os.environ["WANDB_MODE"] = "dryrun"


datasets_list = [
                # 'BlipV3Dataset',
                'WaferDataset',
                'Earthquakes'
               ]

models_list = [
                'dnn',
                # 'rnn'
             ]

lr_list = [ 
            # 5e-2,
            5e-4
          ]

sys.path.append('/home/rdoddaiah/work/TimeSeriesSaliencyMaps')
parser = argparse.ArgumentParser(description='NTE Time Series')
parser.add_argument('--lr', type=float, default=5e-4, choices=lr_list, help='learning rate',required=False)
parser.add_argument('--loss', default='ce',help='loss function',required=False)
parser.add_argument('--epochs', default=120,help='Number of Epochs',required=False)
parser.add_argument('--batch_size', default=32,help='Batch Size',required=False)
parser.add_argument('--gpu', default=1,help='GPU usage',required=False)
parser.add_argument('--save_dir', default='/home/ramesh/work/TimeSeriesSaliencyMaps/nte/trained_models/all_bbms',help='dir to save models', required=False)
# parser.add_argument('--save_dir', default='/tmp',help='dir to save models',required=False)
parser.add_argument('--model', default='dnn', choices=models_list, help='Which model.. DNN by default', required=False)
parser.add_argument("-f", "--fff", help="a dummy argument to fool ipython", default="1")
parser.add_argument('--dataset', help='Which all datasets to train', default='WaferDataset', choices=datasets_list, required=False)

args = parser.parse_args()
print("Args",args)

if args.gpu:
    device = 'cpu' if torch.cuda.is_available() else 'cpu'
else:
    device = 'cpu'

# Configuration
# Hyper-parameters

print("Model List:",models_list)
print("Dataset_List:",datasets_list)

config = {
    "Learning Rate ": lr_list,
    "loss Function": args.loss,
    "Epochs": args.epochs,
    "BatchSize": args.batch_size,
    "BBM save_dir": args.save_dir,
    "BBM models":models_list,
    "BBM datasets":datasets_list,
    "GPU enabled":args.gpu
}


for model in models_list:
    for data_set in datasets_list:

        if data_set == 'BlipV3Dataset':
            dataset = BlipV3Dataset()
        elif data_set == 'WaferDataset':
            dataset = WaferDataset()
        elif data_set == 'EarthquakesDataset':
            dataset = EarthquakesDataset()
        
        for lr in lr_list:
            hyper_params = Munch({
                "model_name": f"{data_set}_model_{model}_loss_{args.loss}_lr_{lr}_epochs_{args.epochs}",
                "model_save_path": args.save_dir+"/",
                "model": model,
                "loss": args.loss,
                "timesteps": len(dataset.train_data[0]),
                "num_classes": 2,
                "rnn_config": {
                    "ninp": 1,
                    "nhid": 20,
                    "nlayers": 1,
                    "nclasses": 2
                },
                "dnn_config": {
                    "layers": [10, 10, 2],
                },
                "batch_size": args.batch_size,
                "num_epochs": args.epochs,
                "learning_rate": args.lr,
                "data_set":data_set,
                "dependency_meta": ""
            })

            print("HYPER PARMS:", hyper_params)

            TIMESTEPS = len(dataset.train_data[0])
            data = dataset.train_data[10]

            print("Data Loaded . for model ", hyper_params.model_name)
            print(f"Train -      Data: {dataset.train_data.shape} | Label: {dataset.train_label.shape}")
            print(f"Test -       Data: {dataset.test_data.shape} | Label: {dataset.test_data.shape}")
            print(f"TIMESTEPS -  Data: {TIMESTEPS}")

            if ENABLE_WANDB:
                TAG = hyper_params.model_name
                BASE_SAVE_DIR = 'results/0109/'
                EXPERIMENT_NAME = f'{model}-{shortuuid.uuid()}'
                print(f" {model}: Working on model:{hyper_params.model_name} dataset: {data_set} ") 
                SAVE_DIR = f'{BASE_SAVE_DIR}/{EXPERIMENT_NAME}'
                os.system(f"mkdir -p {SAVE_DIR}")
                os.system(f"mkdir -p ./wandb/{TAG}/")
                wandb.init(entity="xai", project=PROJECT_NAME, name=hyper_params.model_name,tags=TAG, config=config,  reinit=True, force=True, dir=f"./wandb/{TAG}/")

            if ENABLE_WANDB:
                wandb.log(hyper_params)
                # wandb.save(SAVE_DIR + "/hyper_params.json")

            trained_model = train_black_box_model(dataset=dataset, hyper_params=hyper_params).to(device)
