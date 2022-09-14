# -*- coding: utf-8 -*-
"""
| **@created on:** 11/5/20,
| **@author:** prathyushsp,
| **@version:** v0.0.1
|
| **Description:**
|
|
| **Sphinx Documentation Status:**
"""
from nte.models.saliency_model import Saliency
import numpy as np
import torch
from nte.utils.perturbation_manager import PerturbationManager
from nte.experiment.evaluation import run_evaluation_metrics


class RandomSaliency(Saliency):

    def __init__(self, background_data, background_label, predict_fn, args, max_itr=1):
        super().__init__(background_data=background_data, background_label=background_label, predict_fn=predict_fn)
        self.confidences = []
        self.perturbations = []
        self.max_itr = max_itr
        self.args = args
        self.softmax_fn = torch.nn.Softmax(dim=-1)

    def generate_saliency(self, data, label, **kwargs):

        category = np.argmax(kwargs['target'].cpu().data.numpy())
        self.perturbation_manager = PerturbationManager(
            original_signal=data.flatten(),
            algo="random", prediction_prob=np.max(kwargs['target'].cpu().data.numpy()), original_label=label,
            sample_id=self.args.single_sample_id)
        for i in range(self.max_itr):
            saliency = np.random.random(len(data.flatten()))
            perturbated_input = torch.tensor(data * saliency, dtype=torch.float32)
            confidence = float(self.softmax_fn(self.predict_fn(perturbated_input))[0][category].item())
            print(f"Generating random saliency  - Itr: {i} | Confidence: {confidence:.4f}")

            if kwargs['save_perturbations'] and not self.args.run_eval_every_epoch:
                self.perturbation_manager.add_perturbation(
                    perturbation=perturbated_input.cpu().detach().numpy().flatten(),
                    step=i, confidence=confidence, saliency=saliency)

            if self.args.run_eval_every_epoch:
                metrics = {
                    'eval_metrics': run_evaluation_metrics(self.args.eval_replacement, kwargs['dataset'],
                                                           data.flatten(),
                                                           self.predict_fn, saliency,
                                                           kwargs['save_dir'], False)}

                if kwargs['save_perturbations']:
                    self.perturbation_manager.add_perturbation(perturbation=perturbated_input.flatten(),
                                                               step=i, confidence=confidence, saliency=saliency,
                                                               insertion=metrics['eval_metrics']['Insertion']['trap'],
                                                               deletion=metrics['eval_metrics']['Deletion']['trap'],
                                                               final_auc=metrics['eval_metrics']['Final']['AUC'],
                                                               saliency_sum=float(np.sum(saliency)),
                                                               saliency_var=float(np.var(saliency))
                                                               )
        return saliency, self.perturbation_manager, []
