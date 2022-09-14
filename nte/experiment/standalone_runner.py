# -*- coding: utf-8 -*-
"""
| **@created on:** 1/13/21,
| **@author:** prathyushsp,
| **@version:** v0.0.1
|
| **Description:**
| 
|
| **Sphinx Documentation Status:** 
"""

from nte.experiment.utils import number_to_dataset, set_global_seed, replacement_sample_config
from nte.models.saliency_model.nte_dual_replacement import NTEDualReplacementGradientSaliency
from nte.experiment.utils import number_to_dataset, set_global_seed, replacement_sample_config
from nte.experiment.default_args import parse_arguments
import torch
from nte.experiment.utils import get_model, dataset_mapper, backgroud_data_configuration, get_run_configuration
from nte.experiment.evaluation import run_evaluation_metrics
import numpy as np

def explain(data, label, method, dataset, model=None, args=None, evaluate=False):
    if args is None:
        args = parse_arguments(True)[0]

    if model is None:
        model = get_model(dest_path="default", dataset=dataset.name, use_cuda=False)

    bg_data = dataset.test_data
    bg_label = dataset.test_label

    set_global_seed(args.seed_value)
    target = torch.nn.Softmax(dim=-1)(model(torch.tensor(data, dtype=torch.float32)))
    explainer = method(background_data=bg_data, background_label=bg_label,
                       predict_fn=model,
                       enable_wandb=False, args=args, use_cuda=False)

    mask, perturbation_manager = explainer.generate_saliency(
        data=data, label=label,
        save_dir='/tmp/', save_perturbations=False, target=target, dataset=dataset)

    print(f"Len: {len(mask)} | Min: {np.min(mask):.4f} | Max: {np.max(mask):.4f} |  Var: {np.var(mask):.4f}")
    print(mask)
    run_evaluation_metrics(args.eval_replacement, dataset, data, model, mask, SAVE_DIR='/tmp', ENABLE_WANDB=False, multi_process=False)
    return mask
