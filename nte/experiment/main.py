# -*- coding: utf-8 -*-
"""
| **@created on:** 8/29/20,
| **@author:** prathyushsp,
| **@version:** v0.0.1
|
| **Description:**
| Run Command :
| python3 main.py mse wafer dnn WaferDataset_model_dnn_loss_ce_lr_0.0005_epochs_200.ckpt  1
|
| **Sphinx Documentation Status:**
|
tasks = 4
samples_per_task = 3
data = list(range(12))

total_loops = 0

for TASK_ID in range(tasks):
    for e, s in enumerate(data[
                          int(TASK_ID) * samples_per_task: int(TASK_ID) * samples_per_task + samples_per_task]):
        cur_ind = e + (int(TASK_ID) * samples_per_task)
        print(f"Task: {TASK_ID}, Index: {cur_ind}, Data: {s}")
        total_loops+=1

print(f"Total Runs: {total_loops}")
|
|
|
"""
import torch
import numpy as np
import os
import json
import wandb
import ssl
from nte.experiment.utils import get_model, dataset_mapper, backgroud_data_configuration, get_run_configuration
import shortuuid
import matplotlib.pyplot as plt
from nte.models.saliency_model import SHAPSaliency, LimeSaliency
from nte.models.saliency_model.rise_saliency import RiseSaliency
from nte.models.saliency_model.cm_gradient_saliency import CMGradientSaliency
from nte.models.saliency_model.random_saliency import RandomSaliency
from nte.models.saliency_model.rise_w_replacement import RiseWReplacementSaliency
from nte.models.saliency_model.pert_multi_class_explainer import PertMultiClassSaliency
from nte.models.saliency_model.dynamask import Dynamask
from nte.models.saliency_model.lmux_explainer import LMUXExplainer
from nte.models.saliency_model.lmux_explainer_v1 import LMUXExplainerV1
from nte.models.saliency_model.distinct_paper import distinct_paper_saliency
from nte.models.saliency_model.pert_explainer import PertSaliency
from nte.models.saliency_model.tsmule.ts_mule import TSMuleSaliency
from nte.models.saliency_model.timeshap import TimeShap
import random
from nte.experiment.evaluation import run_evaluation_metrics
from nte.experiment.default_args import parse_arguments
import seaborn as sns
from nte.experiment.utils import number_to_dataset, set_global_seed, replacement_sample_config
from nte.utils import CustomJsonEncoder

#todo: Prathyush - Fix raw imports
from nte.data.real.univariate.multi_class.MixedShapes.MixedShapesModel import MixedShapesCNNModel

sns.set_style("darkgrid")

ssl._create_default_https_context = ssl._create_unverified_context
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

np.set_printoptions(suppress=True)
np.set_printoptions(precision=4)
torch.set_printoptions(precision=4)

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor

ENABLE_WANDB = True
WANDB_DRY_RUN = False

BASE_SAVE_DIR = 'results_v1/2312/'
# BASE_SAVE_DIR = '/tmp/'
if WANDB_DRY_RUN:
    os.environ["WANDB_MODE"] = "dryrun"

if __name__ == '__main__':
    # torch.set_printoptions(sci_mode=True)
    args = parse_arguments()
    print("Config: \n", json.dumps(args.__dict__, indent=2))

    if args.dataset in number_to_dataset.keys():
        args.dataset = number_to_dataset[args.dataset]

    if args.enable_seed:
        set_global_seed(args.seed_value)

    ENABLE_SAVE_PERTURBATIONS = args.save_perturbations
    PROJECT_NAME = args.pname
    dataset = dataset_mapper(DATASET=args.dataset)

    TAG = f'{args.algo}-{args.dataset}-{args.background_data}-{args.background_data_perc}-run-{args.run_id}'
    BASE_SAVE_DIR = BASE_SAVE_DIR + "/" + TAG

    # todo Ramesh: Load black box model -> check this in utils.py
    model = get_model(dest_path=args.bbm_path, dataset=args.dataset,
                      use_cuda=use_cuda, bbm=args.bbm, multi_class=args.multi_class)
    softmax_fn = torch.nn.Softmax(dim=-1)

    bg_data, bg_len = backgroud_data_configuration(BACKGROUND_DATA=args.background_data,
                                                   BACKGROUND_DATA_PERC=args.background_data_perc,
                                                   dataset=dataset)

    print(f"Using {args.background_data_perc}% of background data. Samples: {bg_len}")

    config = args.__dict__

    explainer = None

    if args.algo == 'cm':
        explainer = CMGradientSaliency(background_data=bg_data[:bg_len], background_label=bg_data[:bg_len],
                                       predict_fn=model, enable_wandb=ENABLE_WANDB, args=args, use_cuda=use_cuda)
    elif args.algo == 'pert':
        explainer = PertSaliency(background_data=bg_data[:bg_len], background_label=bg_data[:bg_len], predict_fn=model,
                                 enable_wandb=ENABLE_WANDB, args=args, use_cuda=use_cuda)
    elif args.algo == 'pert-mc':
        explainer = PertMultiClassSaliency(background_data=bg_data[:bg_len], background_label=bg_data[:bg_len],
                                           predict_fn=model,
                                           enable_wandb=ENABLE_WANDB, args=args, use_cuda=use_cuda)
    elif args.algo == 'dyna':
        explainer = Dynamask(background_data=bg_data[:bg_len], background_label=bg_data[:bg_len],
                                           predict_fn=model,
                                           enable_wandb=ENABLE_WANDB, args=args, use_cuda=use_cuda)
    elif args.algo == 'lmux':
        explainer = LMUXExplainer(background_data=bg_data[:bg_len], background_label=bg_data[:bg_len],
                                  predict_fn=model, enable_wandb=ENABLE_WANDB, args=args, use_cuda=use_cuda)
    elif args.algo == 'lmux_v1':
        explainer = LMUXExplainerV1(background_data=bg_data[:bg_len], background_label=bg_data[:bg_len],
                                  predict_fn=model, enable_wandb=ENABLE_WANDB, args=args, use_cuda=use_cuda)
    elif args.algo == 'distinct':
        explainer = distinct_paper_saliency(background_data=bg_data[:bg_len], use_cuda=use_cuda,
                                            background_label=bg_data[:bg_len],
                                            predict_fn=model, enable_wandb=ENABLE_WANDB, args=args)
    elif args.algo == 'lime':
        explainer = LimeSaliency(
            background_data=bg_data[:bg_len], background_label=bg_data[:bg_len], predict_fn=model,
            args=args)
    elif args.algo == 'tsmule':
        explainer = TSMuleSaliency(
            background_data=bg_data[:bg_len], background_label=bg_data[:bg_len], predict_fn=model,
            args=args)
    elif args.algo == 'timeshap':
        explainer = TimeShap(
            background_data=bg_data[:bg_len], background_label=bg_data[:bg_len], predict_fn=model,
            args=args)
    elif args.algo == 'shap':
        explainer = SHAPSaliency(
            background_data=bg_data[:bg_len], background_label=bg_data[:bg_len], predict_fn=model, args=args)
    elif args.algo == 'rise':
        # config["num_masks"] = 10000
        config["num_masks"] = args.num_masks
        explainer = RiseSaliency(background_data=bg_data[:bg_len], background_label=bg_data[:bg_len],
                                 predict_fn=model, args=args,
                                 num_masks=config['num_masks'])
    elif args.algo == 'random':
        explainer = RandomSaliency(
            background_data=bg_data[:bg_len], background_label=bg_data[:bg_len], predict_fn=model, args=args, max_itr=config['num_masks'])
    elif args.algo == 'rise-rep':
        # config["num_masks"] = 10000
        config["num_masks"] = args.num_masks
        explainer = RiseWReplacementSaliency(background_data=bg_data[:bg_len], background_label=bg_data[:bg_len],
                                             predict_fn=model,
                                             args=args, num_masks=config['num_masks'])
    config = {**config, **{
        "tag": TAG,
        "algo": args.algo,
    }}

    dataset_len = len(dataset.test_data)

    ds = get_run_configuration(args=args, dataset=dataset, TASK_ID=args.task_id)

    for ind, (original_signal, original_label) in ds:
        try:
            if args.enable_seed_per_instance:
                set_global_seed(random.randint())
            metrics = {'epoch': {}}
            cur_ind = args.single_sample_id if args.run_mode == 'single' else (
                ind + (int(args.task_id) * args.samples_per_task))
            UUID = dataset.valid_name[cur_ind] if args.dataset_type == 'valid' else shortuuid.uuid()
            EXPERIMENT_NAME = f'{args.algo}-{cur_ind}-R{args.run_id}-RT{args.r_index}-{UUID}-C{ind}-T{args.task_id}-S{args.samples_per_task}-TS{(int(args.task_id) * args.samples_per_task)}-TT{(ind+int(args.task_id) * args.samples_per_task)}'
            print(
                f" {args.algo}: Working on dataset: {args.dataset} index: {cur_ind} [{((cur_ind + 1) / dataset_len * 100):.2f}% Done]")
            SAVE_DIR = f'{BASE_SAVE_DIR}/{EXPERIMENT_NAME}'
            os.system(f'mkdir -p "{SAVE_DIR}"')
            os.system(f'mkdir -p "./wandb/{TAG}/"')
            config['save_dir'] = SAVE_DIR

            if args.run_mode == 'single' and args.dynamic_replacement == False:
                config = {**config, **replacement_sample_config(
                    xt_index=args.single_sample_id, rt_index=args.r_index, dataset=dataset, model=model, dataset_type=args.background_data)}

            json.dump(config, open(SAVE_DIR + "/config.json", 'w'), indent=2, cls=CustomJsonEncoder)
            if ENABLE_WANDB:
                wandb.init(entity="xai", project=PROJECT_NAME, name=EXPERIMENT_NAME, tags=TAG,
                           config=config, reinit=True, force=True, dir=f"./wandb/{TAG}/")
                # plt.plot(original_signal, label="Original Signal")
                # plt.xlabel("Timesteps")
                # plt.ylabel("Values")

            original_signal = torch.tensor(original_signal, dtype=torch.float32)

            with torch.no_grad():
                if args.bbm == 'rnn':
                    if args.bbmg == 'yes':
                        timesteps = int(dataset.train_class_0_data[0].shape[0]/args.window_size)
                        target = softmax_fn(model(original_signal.reshape(-1, timesteps, args.window_size)))
                    else:
                        target = softmax_fn(model(original_signal.reshape(1, -1)))

                    if args.multi_class == 'yes':
                        target = target[0]
                elif args.bbm == 'dnn':
                    target = softmax_fn(model(original_signal))
                elif args.bbm == "cnn":
                    target = softmax_fn(model(original_signal))[0]
                else:
                    raise Exception(f"Black Box model not supported: {args.bbm}")

            category = np.argmax(target.cpu().data.numpy())
            args.dataset = dataset
            if ENABLE_WANDB:
                wandb.run.summary[f"prediction_class"] = category
                wandb.run.summary[f"prediction_prob"] = np.max(target.cpu().data.numpy())
                wandb.run.summary[f"label"] = original_label
                wandb.run.summary[f"target"] = target.cpu().data.numpy()

            if args.background_data == "none":
                explainer.background_data = original_signal
                explainer.background_label = original_label

            explainer.timesteps = int(dataset.train_class_0_data[0].shape[0]/args.window_size)

            mask, perturbation_manager, n_masks = explainer.generate_saliency(
                data=original_signal.reshape([1, -1]).cpu().detach().numpy(), label=original_label,
                save_dir=SAVE_DIR, save_perturbations=args.save_perturbations, target=target, dataset=dataset)

            mask = mask.flatten()

            num = mask-mask.min()
            den = mask.max()-mask.min()

            if(den != 0):
                mask = (mask-mask.min())/(mask.max()-mask.min())

            classes = len(n_masks)
            for i in range(classes):
                num = (n_masks[i]-n_masks[i].min())
                den = (n_masks[i].max()-n_masks[i].min())
                if(den != 0):
                    n_masks[i] = num/den


            # Evaluation Metrics
            if args.run_eval:
                metrics['eval_metrics'] = run_evaluation_metrics(args.eval_replacement, dataset, original_signal, model,
                                                                 mask, n_masks, classes,category, SAVE_DIR, ENABLE_WANDB, main_args=args)
                if ENABLE_WANDB:
                    for mk, mv in metrics['eval_metrics'].items():
                        for k, v in mv.items():
                            if isinstance(v, list):
                                pass
                            else:
                                wandb.run.summary[f"{mk} {k}"] = v
                    wandb.run.summary["Saliency Sum Final"] = np.sum(mask)
                    wandb.run.summary["Saliency Var Final"] = np.var(mask)

                json.dump(metrics, open(SAVE_DIR + "/metrics.json", 'w'),
                          indent=2)

            if args.save_perturbations:
                perturbation_manager.to_csv(
                    SAVE_DIR=SAVE_DIR, TAG=TAG, UUID=UUID, SAMPLE_ID=cur_ind)
                if ENABLE_WANDB:
                    wandb.save(
                        f"{SAVE_DIR}/perturbations-{TAG}-{UUID}-{cur_ind}.csv")
        except Exception as e:
            with open(f'/tmp/{TAG}_error.log', 'a+') as f:
                f.write(e)
                f.write(e.__str__())
                f.write("\n\n")
