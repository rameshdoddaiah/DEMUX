from abc import ABCMeta, abstractmethod
import torch
import shap
import lime
import lime.lime_tabular
import numpy as np
import time
from torch import nn
from nte.utils.perturbation_manager import PerturbationManager
from nte.experiment.utils import save_timeseries


class Saliency(metaclass=ABCMeta):
    def __init__(self, background_data, background_label, predict_fn):
        self.background_data = background_data
        self.background_label = background_label
        self.predict_fn = predict_fn
        self.perturbation_manager = None
        pass

    @abstractmethod
    def generate_saliency(self, data, label, **kwargs):
        pass


class SHAPSaliency(Saliency):
    class SHAPHelper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.perturbations = None
            self.confidences = []

        def forward(self, x):
            if self.perturbations is None:
                self.perturbations = x
            else:
                np.concatenate([self.perturbations, x])
            x = torch.tensor(x, dtype=torch.float32)
            res = torch.softmax(self.model(x), dim=-1).cpu().detach().numpy()
            self.confidences.append(np.max(res))
            return res

    def __init__(self, background_data, background_label, predict_fn, args):
        super().__init__(predict_fn=predict_fn, background_data=background_data, background_label=background_label)
        self.fetch_class = self.SHAPHelper(self.predict_fn)
        # fetch_class = lambda d: self.predict_fn(d)[2]
        self.explainer = shap.KernelExplainer(model=self.fetch_class, data=self.background_data)
        self.args = args

    def generate_saliency(self, data, label, **kwargs):
        shap_values = self.explainer.shap_values(data)
        if kwargs['save_perturbations']:
            self.perturbation_manager = PerturbationManager(
                original_signal=data.flatten(),
                algo="lime", prediction_prob=np.max(kwargs['target'].cpu().data.numpy()), original_label=label)
            self.perturbation_manager.update_perturbation(self.fetch_class.perturbations,
                                                          confidences=self.fetch_class.confidences)

        top_label  = np.argmax(kwargs['target'].cpu().data.numpy())
        mask = shap_values[top_label]
        norm_mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))

        save_timeseries(mask=norm_mask, raw_mask=norm_mask, time_series=data,
                        save_dir=kwargs['save_dir'],
                        algo='shap', dataset=kwargs['dataset'], category=label)

        classes = kwargs['target'].cpu().data.numpy().shape[0]
        for c in range(classes):
            shap_values[c] = shap_values[c][0]
        # print(f"Saliency Sum: {np.sum(norm_mask)} | Saliency Var: {np.var(norm_mask)}")
        return norm_mask, self.perturbation_manager, shap_values


# class SHAPSaliency(Saliency):
#
#     class SHAPHelper(nn.Module):
#         def __init__(self, model):
#             super().__init__()
#             self.model = model
#
#         def forward(self,x):
#             return torch.softmax(self.model(x), dim=1)
#
#     def __init__(self, data, label , model, **kwargs):
#         super().__init__(predict_fn=model, data=data, label=label)
#         helper = self.SHAPHelper(self.predict_fn)
#         self.explainer = shap.GradientExplainer(model=helper, data=torch.tensor(self.data, dtype=torch.float32))
#
#     def generate_saliency(self, data, label):
#         shap_values = self.explainer.shap_values(torch.tensor(data, dtype=torch.float32))
#         return shap_values[0]


class LimeSaliency(Saliency):
    class LimeHelper(nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model
            self.perturbations = None
            self.confidences = []

        def forward(self, x):
            if self.perturbations is None:
                self.perturbations = x
            else:
                np.concatenate([self.perturbations, x])
            x = torch.tensor(x, dtype=torch.float32)
            res = torch.softmax(self.model(x), dim=-1).cpu().detach().numpy()
            self.confidences.append(np.max(res))
            return res

    def __init__(self, background_data, background_label, predict_fn, args):
        super().__init__(background_data=background_data, background_label=background_label, predict_fn=predict_fn)
        self.lime_explainer = lime.lime_tabular.LimeTabularExplainer(background_data, verbose=False)
        self.args= args

    def generate_saliency(self, data, label, **kwargs):
        data = data.reshape([1, -1])
        timesteps = data.shape[1]
        saliencies = []
        fetch_class = self.LimeHelper(self.predict_fn)
        # fetch_class = lambda d: self.predict_fn(d)[2]
        total_candidates = len(data)
        top_label  = np.argmax(kwargs['target'].cpu().data.numpy())
        classes = kwargs['target'].cpu().data.numpy().shape[0]
        for e, d in enumerate(data):
            start = time.time()
            if self.args.window_size >1:
                # lime_gbr = self.lime_explainer.explain_instance(d.reshape(-1,kwargs['timesteps'],kwargs['window_size']), fetch_class, num_features=timesteps)
                # lime_gbr = self.lime_explainer.explain_instance(d.reshape(-1,kwargs['window_size'],kwargs['timesteps']), fetch_class, num_features=timesteps)
                lime_gbr = self.lime_explainer.explain_instance(d, fetch_class, num_features=timesteps,window_size=kwargs['window_size'],timesteps=kwargs['timesteps'])
            else:
                lime_gbr = self.lime_explainer.explain_instance(d, fetch_class, top_labels=classes, num_features=timesteps)
                # lime_gbr = self.lime_explainer.explain_instance(d, fetch_class, num_features=timesteps)
            n_masks = [] * classes
            for c in range(classes):
                lime_values = np.zeros(timesteps)
                for ids, val in lime_gbr.local_exp[c]:  # np.argmax(fetch_class(d))
                    lime_values[ids] = val
                n_masks.append(np.array(lime_values))
            saliencies.append(np.array(n_masks[top_label]))
            print(f"[{e / total_candidates * 100:.2f}%] Candidate {e} took {time.time() - start:.4f}s")

        if kwargs['save_perturbations']:
            self.perturbation_manager = PerturbationManager(
                original_signal=data.flatten(),
                algo="lime", prediction_prob=np.max(kwargs['target'].cpu().data.numpy()), original_label=label)
            self.perturbation_manager.update_perturbation(fetch_class.perturbations,
                                                          confidences=fetch_class.confidences)
        mask = np.array(saliencies).flatten()
        norm_mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask))
        save_timeseries(mask=norm_mask, raw_mask=mask, time_series=data.flatten(),
                        save_dir=kwargs['save_dir'],
                        algo='lime', dataset=kwargs['dataset'], category=label)

        # print(f"Saliency Sum: {np.sum(norm_mask)} | Saliency Var: {np.var(norm_mask)}")
        return norm_mask, self.perturbation_manager, n_masks


from nte.models.saliency_model.nte_explainer import NTEGradientSaliency
from nte.models.saliency_model.nte_dual_replacement import NTEDualReplacementGradientSaliency
from nte.models.saliency_model.rise_saliency import RiseSaliency
