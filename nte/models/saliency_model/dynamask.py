# -*- coding: utf-8 -*-
"""
"""


import torch
import cv2
import sys
import numpy as np
import os
import json
import wandb
import pandas as pd
import ssl
from nte.experiment.utils import get_image, tv_norm, \
    save, numpy_to_torch, load_model, get_model, send_plt_to_wandb, save_timeseries, dataset_mapper, \
    backgroud_data_configuration, get_run_configuration
from nte.experiment.evaluation import qm_plot
import tqdm
import shortuuid
import matplotlib.pyplot as plt
from nte.models.saliency_model import SHAPSaliency, LimeSaliency
from nte.models.saliency_model.rise_saliency import RiseSaliency
from nte.models.saliency_model.cm_gradient_saliency import CMGradientSaliency
from scipy.ndimage.filters import gaussian_filter1d
from scipy.ndimage.filters import median_filter
import logging
import random
from nte.experiment.evaluation import run_evaluation_metrics
import seaborn as sns
from nte.utils.perturbation_manager import PerturbationManager
from nte.experiment.softdtw_loss_v1 import SoftDTW
from nte.models.saliency_model import Saliency
from torch.optim.lr_scheduler import ExponentialLR, StepLR
import math
from nte.utils.priority_buffer import PrioritizedBuffer
import scipy
import torch
import torch.nn as nn
import torch.optim as optim

from tslearn.clustering import TimeSeriesKMeans
from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesResampler
from numpy import mean


class Dynamask(Saliency):
    def __init__(self, background_data, background_label, predict_fn, enable_wandb, use_cuda, args):
        super(Dynamask, self).__init__(background_data=background_data,
                                                     background_label=background_label,
                                                     predict_fn=predict_fn)
        self.enable_wandb = enable_wandb
        self.use_cuda = use_cuda
        self.args = args
        self.softmax_fn = torch.nn.Softmax(dim=-1)
        self.perturbation_manager = None
        self.r_index = random.randrange(0, len(self.background_data)) if self.args.r_index < 0 else self.args.r_index
        self.rs_priority_buffer = None
        self.ro_priority_buffer = None
        self.eps = 1.0
        self.eps_decay = 0.9991

    def generate_saliency(self, data, label, **kwargs):

        if isinstance(data, np.ndarray):
            data = torch.tensor(data.flatten(), dtype=torch.float32)

        reached_perturbs = 1
        orig_label = label
        flag = 0
        orig_pred_prob = (kwargs['target'].cpu().data.numpy())
        category = np.argmax(kwargs['target'].cpu().data.numpy())
        orig_sorted_category = np.argsort(kwargs['target'].cpu().data.numpy())
        ordered_orig_sorted_category = np.argsort(kwargs['target'].cpu().data.numpy()*-1)
        num_classes = getattr(kwargs['dataset'], f"num_classes")

        categories = ordered_orig_sorted_category.tolist()
        top_class = categories[0]

        label_category_pred_prob = kwargs['target'].cpu().data.numpy()[label]

        top_n_masks = []
        top_n_various_perturbations = []
        top_n_norm_masks = []
        top_n_prob_norm_masks = []

        top_n_range = 1

        for top_n in range(top_n_range):

            various_perturbations = []
            various_perturbations_labels = []
            various_perturbations_masks = []
            reached_perturbs = 0

            self.eps = 1.0
            if kwargs['save_perturbations']:
                self.perturbation_manager = PerturbationManager(original_signal=data.cpu().detach().numpy().flatten(), algo=self.args.algo, prediction_prob=np.max(kwargs['target'].cpu().data.numpy()),
                                                                original_label=label, sample_id=self.args.single_sample_id)

            plt.plot(data, label="Original Signal Norm")
            gkernel = cv2.getGaussianKernel(3, 0.5)
            gaussian_blur_signal = cv2.filter2D(data.cpu().detach().numpy(), -1, gkernel).flatten()
            plt.plot(gaussian_blur_signal, label="Gaussian Blur")
            median_blur_signal = median_filter(data, 3)
            plt.plot(median_blur_signal, label="Median Blur")
            blurred_signal = (gaussian_blur_signal + median_blur_signal) / 2
            plt.plot(blurred_signal, label="Blurred Signal")
            mask_init = np.full(shape=len(data), fill_value=1e-2)
            blurred_signal_norm = blurred_signal / np.max(blurred_signal)
            plt.plot(blurred_signal_norm, label="Blur Norm")

            if self.enable_wandb:
                wandb.log({'Initialization': plt}, step=0)

            blurred_signal = torch.tensor((blurred_signal_norm.reshape(1, -1)), dtype=torch.float32)
            n_mask = []
            for i in range(num_classes):
                mask_init = np.random.uniform(size=len(data), low=-1e-2, high=1e-2)
                n_mask.append(numpy_to_torch(mask_init, use_cuda=self.use_cuda))

            optimizer = torch.optim.Adam([n_mask[c] for c in range(num_classes)], lr=1e-4, weight_decay=1e-3)

            print(f"{self.args.algo}: Category with highest probability {category}")
            print(f"{self.args.algo}: Optimizing.. ")

            metrics = {"TV Norm": [], 'TV Coeff': [], "MSE": [], "Budget": [], "Total Loss": [], "CS Loss":[],
                       "Confidence": [],
                       "Top Confidence": [],
                       "Saliency Var": [],
                       "Saliency Sum": [], "MSE Coeff": [], "L1 Coeff": [], "Mean Gradient": [],
                       "Category": [],
                       "Label_pred_prob": [],
                       "epoch": {}, 'MSE Var': [], 'DIST': []}

            metrics['Label_pred_prob'].append(float(label_category_pred_prob.item()))

            mse_loss_fn = torch.nn.MSELoss(reduction='mean')

            for i in (range(self.args.max_itr)):

                CUR_DIR = kwargs['save_dir'] + f'/epoch_{i}/'

                Rt = []
                Zt = torch.mean(data).cpu().detach().numpy()
                Zt = torch.tensor(np.repeat(Zt, data.shape[0]), dtype=torch.float32)
                perturbated_input = []

                for c in range(num_classes):
                    Rt.append(Zt)
                    perturbated_input.append(data.mul(n_mask[c]) + Rt[c].mul(1 - n_mask[c]))

                for c in range(num_classes):
                    perturbated_input[c] = perturbated_input[c].flatten()

                outputs = [[0]*num_classes]*num_classes
                for c in range(num_classes):
                    outputs[c] = self.softmax_fn(self.predict_fn(perturbated_input[c].reshape([1, -1])))[0]
                metrics['Confidence'].append(float(outputs[top_class][top_class].item()))
                metrics['Top Confidence'].append(float(outputs[top_class][top_class].item()))

                if kwargs['save_perturbations'] and not self.args.run_eval_every_epoch:
                    self.perturbation_manager.add_perturbation(perturbation=perturbated_input.cpu(
                    ).detach().numpy().flatten(), step=i, confidence=metrics['Confidence'][-1])
                if self.args.bbm == 'rnn':
                    c_mse = []
                    for c in range(num_classes):
                        c_mse.append(self.args.mse_coeff * mse_loss_fn(outputs[c], kwargs['target']))
                else:
                    c_mse = []
                    for c in range(num_classes):
                        c_mse.append(self.args.mse_coeff * mse_loss_fn(outputs[c], kwargs['target']))

                c_budget = []
                for c in range(num_classes):
                    c_budget.append(self.args.l1_coeff *
                                    torch.mean(torch.abs(n_mask[c])) * float(self.args.enable_budget))

                c_tvnorm = []
                for c in range(num_classes):
                    c_tvnorm.append(self.args.tv_coeff *
                                    tv_norm(n_mask[c], self.args.tv_beta) * float(self.args.enable_tvnorm))

                mask_mse_loss = []
                mse_var = np.var(metrics['MSE']) if len(metrics['MSE']) > 1 else 0.0

                loss = 0
                loss = c_mse[top_class] + c_budget[top_class] + c_tvnorm[top_class]
                optimizer.zero_grad()
                loss.backward()

                metrics['Mean Gradient'].append(float(np.mean(n_mask[top_class].grad.cpu().detach().numpy())))
                metrics['TV Norm'].append(float(c_tvnorm[top_class].item()))
                metrics['MSE'].append(float(c_mse[top_class].item()))
                metrics['Saliency Sum'].append(float(np.sum(n_mask[top_class].cpu().detach().numpy() > 0)))
                metrics['Saliency Var'].append(float(np.var(n_mask[top_class].cpu().detach().numpy())))
                metrics['MSE Coeff'].append(self.args.mse_coeff)
                metrics['TV Coeff'].append(self.args.tv_coeff)
                metrics['L1 Coeff'].append(self.args.l1_coeff)
                metrics['MSE Var'].append(mse_var)


                torch.set_printoptions(precision=2)

                if self.args.top_n == 'yes':
                    reached_prob = outputs[ordered_orig_sorted_category[top_n]]
                    reached_prob_category = ordered_orig_sorted_category[top_n]
                    reached_prob_upto = kwargs['target'].cpu().data.numpy()[ordered_orig_sorted_category[top_n]]
                    multiplier_prob = reached_prob_upto

                if self.args.prob_upto > 0:
                    reached_prob_upto = self.args.prob_upto

                if self.args.print == 'yes':
                    print(
                        f"Iter: {i}/{self.args.max_itr}"
                        f"| {i / self.args.max_itr * 100:.2f}%"
                        f"| Loss : {loss:.4f}"
                        f"| Top_class : {top_class} Reached Prob:{outputs[top_class][top_class]:.4f}"
                        f"| Prob : {kwargs['target'].cpu().data.numpy()[top_class]:.4f}"
                    )

                optimizer.step()

                for c in range(num_classes):
                    n_mask[c].data.clamp_(0,1)

                if self.args.run_eval_every_epoch:
                    m = n_mask[top_class].cpu().detach().numpy().flatten()
                    metrics["epoch"][f"epoch_{i}"] = {'eval_metrics': run_evaluation_metrics(
                        self.args.eval_replacement, kwargs['dataset'], data, self.predict_fn, m, kwargs['save_dir'], False)}

                    if kwargs['save_perturbations']:
                        self.perturbation_manager.add_perturbation(
                            perturbation=perturbated_input.cpu().detach().numpy().flatten(),
                            saliency=n_mask[top_class].cpu().detach().numpy(),
                            step=i, confidence=metrics['Confidence'][-1],
                            insertion=metrics["epoch"][f"epoch_{i}"]['eval_metrics']['Insertion']['trap'],
                            deletion=metrics["epoch"][f"epoch_{i}"]['eval_metrics']['Deletion']['trap'],
                            final_auc=metrics["epoch"][f"epoch_{i}"]['eval_metrics']['Final']['AUC'],
                            mean_gradient=metrics['Mean Gradient'][-1],
                            tv_norm=metrics['TV Norm'][-1],
                            main_loss=metrics["MSE"][-1],
                            budget=metrics["Budget"][-1],
                            total_loss=metrics["Total Loss"][-1],
                            category=metrics["Category"][-1],
                            saliency_sum=metrics["Saliency Sum"][-1],
                            saliency_var=metrics["Saliency Var"][-1])

        various_perturbations_masks_mean = n_mask[top_class].data.numpy()
        mask_t = various_perturbations_masks_mean
        main_saliency = various_perturbations_masks_mean

        np.set_printoptions(suppress=True)
        np.set_printoptions(precision=3)
        main_saliency = main_saliency
        main_saliency = main_saliency.astype(np.float32)
        if self.args.bbm == 'rnn':
            Final_unique_prob_pred = self.softmax_fn(self.predict_fn((data*main_saliency).reshape(1, -1))).unsqueeze(0)
        else:
            Final_unique_prob_pred = self.softmax_fn(self.predict_fn(data*main_saliency))

        upsampled_mask = torch.tensor(main_saliency, dtype=torch.float32)
        mask = torch.tensor(main_saliency, dtype=torch.float32)
        if self.enable_wandb:
            wandb.save(kwargs['save_dir'] + "/metrics.json")
            wandb.save(kwargs['save_dir'] + "/upsampled_mask.npy")
            wandb.save(kwargs['save_dir'] + "/mask.npy")

        mask = mask.squeeze(0).squeeze(0)
        mask = mask.cpu().detach().numpy().flatten()

        if self.enable_wandb:
            wandb.run.summary["pos_features"] = len(np.where(mask > 0)[0])
            wandb.run.summary["neg_features"] = len(np.where(mask < 0)[0])
            wandb.run.summary["pos_sum"] = np.sum(mask[np.argwhere(mask > 0)])
            wandb.run.summary["neg_sum"] = np.sum(mask[np.argwhere(mask < 0)])

        norm_mask = mask


        if self.enable_wandb:
            wandb.run.summary["norm_saliency_sum"] = np.sum(mask)
            wandb.run.summary["norm_saliency_var"] = np.var(mask)
            wandb.run.summary["norm_pos_sum"] = np.sum(norm_mask[np.argwhere(mask > 0)])
            wandb.run.summary["norm_neg_sum"] = np.sum(norm_mask[np.argwhere(mask < 0)])
            wandb.run.summary["label"] = top_class

        np.save("./r_mask", mask)
        np.save("./n_mask", norm_mask)

        new_mask = []
        for i in range(num_classes):
            new_mask.append(n_mask[i].cpu().detach().numpy()[0][0])

        return norm_mask, self.perturbation_manager, new_mask
