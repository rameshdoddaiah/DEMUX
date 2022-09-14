from tsmule.xai.lime import LimeTS


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
from tsmule.xai.lime import LimeTS

class TSMuleSaliency(Saliency):

    def __init__(self, background_data, background_label, predict_fn, args, max_itr=1):
        super().__init__(background_data=background_data, background_label=background_label, predict_fn=predict_fn)
        self.confidences = []
        self.perturbations = []
        self.max_itr = max_itr
        self.args = args
        self.softmax_fn = torch.nn.Softmax(dim=-1)
        self.explainer = LimeTS()

    def generate_saliency(self, data, label, **kwargs):
        classes = len(kwargs['target'].cpu().data.numpy())
        category = np.argmax(kwargs['target'].cpu().data.numpy())
        n_masks = []*classes
        for i in range(classes):
            pred_fn = lambda x: float(self.softmax_fn(self.predict_fn(torch.tensor(x.T, dtype=torch.float32)))[0][i].item())
            saliencies = self.explainer.explain(x=data.reshape([-1, 1]),
                                                predict_fn=pred_fn,
                                                segmentation_method="uniform",
                            )
            n_masks.append(saliencies.flatten())
        mask = np.array(n_masks[category]).flatten()
        norm_mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))

        if kwargs['save_perturbations']:
            self.perturbation_manager = PerturbationManager(
                original_signal=data.flatten(),
                algo="lime", prediction_prob=np.max(kwargs['target'].cpu().data.numpy()), original_label=label)
            self.perturbation_manager.update_perturbation(fetch_class.perturbations,
                                                          confidences=fetch_class.confidences)
        return norm_mask, self.perturbation_manager, n_masks
