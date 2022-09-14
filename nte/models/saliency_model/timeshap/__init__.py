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
import timeshap.utils as utils
import timeshap.explainer as timeshap_exp
import pandas as pd


class NumericalNormalizer:
    def __init__(self, fields: list):
        self.metrics = {}
        self.fields = fields

    def fit(self, df: pd.DataFrame) -> list:
        means = df[self.fields].mean()
        std = df[self.fields].std()
        for field in self.fields:
            field_mean = means[field]
            field_stddev = std[field]
            self.metrics[field] = {'mean': field_mean, 'std': field_stddev}

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        # Transform to zero-mean and unit variance.
        for field in self.fields:
            f_mean = self.metrics[field]['mean']
            f_stddev = self.metrics[field]['std']
            # OUTLIER CLIPPING to [avg-3*std, avg+3*avg]
            df[field] = df[field].apply(lambda x: f_mean - 3 * f_stddev if x < f_mean - 3 * f_stddev else x)
            df[field] = df[field].apply(lambda x: f_mean + 3 * f_stddev if x > f_mean + 3 * f_stddev else x)
            if f_stddev > 1e-5:
                df[f'p_{field}_normalized'] = df[field].apply(lambda x: ((x - f_mean)/f_stddev))
            else:
                df[f'p_{field}_normalized'] = df[field].apply(lambda x: x * 0)
        return df


class TimeShap(Saliency):

    def __init__(self, background_data, background_label, predict_fn, args, max_itr=1):
        super().__init__(background_data=background_data, background_label=background_label, predict_fn=predict_fn)
        self.confidences = []
        self.perturbations = []
        self.max_itr = max_itr
        self.args = args
        self.softmax_fn = torch.nn.Softmax(dim=-1)

        bg_df = pd.DataFrame(self.background_data)
        raw_model_features = bg_df.columns.tolist()
        model_features = [f"p_{x}_normalized" for x in raw_model_features]
        self.feature_dict = {'rs': 1, 'nsamples': 15000, 'feature_names': model_features}
        normalizor = NumericalNormalizer(raw_model_features)
        normalizor.fit(bg_df)
        d_train_normalized = normalizor.transform(bg_df)
        self.baseline = utils.calc_avg_event(d_train_normalized, numerical_feats=model_features, categorical_feats=[])

    def generate_saliency(self, data, label, **kwargs):
        classes = len(kwargs['target'].cpu().data.numpy())
        category = np.argmax(kwargs['target'].cpu().data.numpy())
        n_masks = []*classes

        def predict_fn(x):
            predictions = self.softmax_fn(self.predict_fn(torch.tensor(x.reshape([x.shape[0], x.shape[-1]]), dtype=torch.float32)))
            return_scores = []
            return_scores.append(predictions.numpy())
            return np.concatenate(tuple(return_scores), axis=0)

        vals = timeshap_exp.local_feat(f=predict_fn,
                                       data = data.reshape([1, 1, -1]),
                                       feature_dict=self.feature_dict,
                                       entity_uuid=-1,  # Not used
                                       entity_col="all_id",  # Not used
                                       baseline=self.baseline,
                                       pruned_idx=0)
        n_masks = np.array(vals["Shapley Value"].tolist()).T
        mask = np.array(n_masks[category]).flatten()
        norm_mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))

        if kwargs['save_perturbations']:
            self.perturbation_manager = PerturbationManager(
                original_signal=data.flatten(),
                algo="lime", prediction_prob=np.max(kwargs['target'].cpu().data.numpy()), original_label=label)
            self.perturbation_manager.update_perturbation(fetch_class.perturbations,
                                                          confidences=fetch_class.confidences)
        return norm_mask, self.perturbation_manager, n_masks
