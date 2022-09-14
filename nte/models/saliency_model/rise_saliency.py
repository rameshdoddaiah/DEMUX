# -*- coding: utf-8 -*-
"""
| **@created on:** 9/19/20,
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


class RiseSaliency(Saliency):

    def __init__(self, background_data, background_label, predict_fn, num_masks: int, args):
        super().__init__(background_data=background_data, background_label=background_label, predict_fn=predict_fn)
        self.num_masks = num_masks
        self.softmax_fn = torch.nn.Softmax(dim=1)
        self.confidences = []
        self.perturbations = []
        self.args = args

    def generate_saliency(self, data, label, **kwargs):
        MASKS = torch.randint(0, 2, (self.num_masks, data.shape[1]), dtype=torch.float)  # Random mask
        top_label  = np.argmax(kwargs['target'].cpu().data.numpy())
        classes = kwargs['target'].cpu().data.numpy().shape[0]
        n_masks = [] * classes
        for c in range(classes):
            outer_accuracy = np.zeros((data.shape[1]))
            count_mask = np.zeros((data.shape[1]))
            for mask in MASKS:
                count_mask += mask.cpu().numpy()
                X_batch_masked = mask * data
                self.perturbations.append(X_batch_masked.cpu().detach().numpy())
                # X_batch_masked = torch.reshape(torch.tensor(X_batch_masked, dtype=torch.float32), (152, 1, 1))
                # if args.bbmg == 'yes':
                #     res = self.softmax_fn(self.predict_fn(X_batch_masked.reshape(-1,11,25)))
                # else:
                if self.args.window_size > 1:
                    res = self.softmax_fn(self.predict_fn(X_batch_masked.reshape(-1,kwargs['timesteps'],kwargs['window_size'])))
                else:
                    res = self.softmax_fn(self.predict_fn(X_batch_masked))
                # predictions = res[0][c].item()
                # self.confidences.append((res.cpu().detach().numpy()[0]))
                predictions = torch.argmax(res).item()
                self.confidences.append(np.max(res.cpu().detach().numpy()))
                # outer_accuracy += (1 * mask if predictions == label else 0 * mask).cpu().detach().numpy()
                outer_accuracy += (1 * mask if predictions == c else 0 * mask).cpu().detach().numpy()
            saliency = outer_accuracy / count_mask
            n_masks.append(np.array(saliency))
        saliency = n_masks[top_label]
        if kwargs['save_perturbations']:
            self.perturbation_manager = PerturbationManager(
                original_signal=data.flatten(),
                algo="lime", prediction_prob=np.max(kwargs['target'].cpu().data.numpy()), original_label=label)
            self.perturbation_manager.update_perturbation(self.perturbations,
                                                          confidences=self.confidences)
        top_class_mask = n_masks[top_label]
        distinct_mask = 0
        for c in range(classes):
            if c != top_label:
                distinct_mask += (top_class_mask - n_masks[c])
        dmin, dmax = np.min(distinct_mask), np.max(distinct_mask)
        den = dmax-dmin
        if den == 0:
            den = 1e-27
        distinct_mask = 2 * ((distinct_mask-dmin)/(den)) - 1
        saliency = distinct_mask
        return saliency, self.perturbation_manager, n_masks