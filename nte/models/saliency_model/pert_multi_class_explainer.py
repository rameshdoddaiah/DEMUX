# -*- coding: utf-8 -*-
"""
| **@created on:** 9/28/20,
| **@author:** prathyushsp,
| **@version:** v0.0.1
|
| **Description:**
|
|
| **Sphinx Documentation Status:**
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
# %matplotlib inline

import scipy
import torch
import torch.nn as nn
# from tensorboardX import SummaryWriter
import torch.optim as optim

from tslearn.clustering import TimeSeriesKMeans
from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, TimeSeriesResampler
from numpy import mean


class LSTMGenerator(nn.Module):
    def __init__(self, in_dim, out_dim, n_layers=1, hidden_dim=256):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layers, batch_first=True)
        self.linear0 = nn.Linear(in_dim, hidden_dim)
        self.linear = nn.Sequential(nn.Linear(hidden_dim, out_dim))
        # self.linear = nn.Sequential(nn.Linear(hidden_dim, out_dim), nn.Tanh())

    def forward(self, input):
        batch_size, seq_len = input.size(0), input.size(1)
        h_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        c_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim)

        # recurrent_features, _ = self.linear0(input,self.hidden_dim)
        recurrent_features, _ = self.lstm(input, (h_0, c_0))
        outputs = self.linear(recurrent_features.contiguous().view(batch_size*seq_len, self.hidden_dim))
        outputs = outputs.view(batch_size, seq_len, self.out_dim)
        return outputs


class LSTMDiscriminator(nn.Module):
    def __init__(self, in_dim, n_layers=1, hidden_dim=256):
        super().__init__()
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(in_dim, hidden_dim, n_layers, batch_first=True)
        self.linear = nn.Sequential(nn.Linear(hidden_dim, 10), nn.Sigmoid())

    def forward(self, input):
        # batch_size, seq_len = input.size(0), input.size(1)
        batch_size, seq_len = 1, input.shape[0]
        h_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        c_0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim)

        recurrent_features, _ = self.lstm(input, (h_0, c_0))
        outputs = self.linear(recurrent_features.contiguous().view(batch_size*seq_len, self.hidden_dim))
        outputs = outputs.view(batch_size, seq_len, 10)
        return outputs


class PertMultiClassSaliency(Saliency):
    def __init__(self, background_data, background_label, predict_fn, enable_wandb, use_cuda, args):
        super(PertMultiClassSaliency, self).__init__(background_data=background_data,
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
        self.eps_decay = 0.9991  # 0.9996#0.997

    def priority_dual_greedy_pick_rt(self, kwargs, data, label):
        self.eps *= self.eps_decay
        if np.random.uniform() < self.eps:
            self.mode = 'Explore'
            # todo: Change Rs and Ro dynamically
            num_classes = getattr(kwargs['dataset'], f"num_classes")
            rs_index = [np.random.choice(len(getattr(self.args.dataset, f"test_class_{int(label)}_data")))]
            Rs, rs_weight = [getattr(self.args.dataset, f"test_class_{int(label)}_data")[rs_index[0]]], [1.0]

            if num_classes > 2:

                min_class = np.argmin(kwargs['target'].cpu().data.numpy())
                max_class = np.argmax(kwargs['target'].cpu().data.numpy())

                if self.args.grad_replacement == 'random_instance':
                    # rand_choice = list(range(0, int(label))) + list(range(int(label)+1, num_classes))
                    # rand_label = np.random.choice(rand_choice)
                    # gen_index = len(getattr(self.args.dataset, f"test_class_{rand_label}_data")) % len(
                    #     getattr(self.args.dataset, f"test_class_{int(label)}_data"))
                    # if gen_index > 0:
                    #     ro_index = [np.random.choice(gen_index)]
                    # else:
                    #     ro_index = [np.random.choice(gen_index+1)]
                    # Ro, ro_weight = [getattr(self.args.dataset, f"test_class_{rand_label}_data")[ro_index[0]]], [1.0]

                    ro_index = [np.random.choice(list(range(0,len(self.ro_priority_buffer.memory))))]
                    Ro, ro_weight = self.ro_priority_buffer.memory[ro_index[0]],[1.0]

                elif self.args.grad_replacement == 'min_class_random_instance':
                    gen_index = len(getattr(self.args.dataset, f"test_class_{min_class}_data")) % len(
                        getattr(self.args.dataset, f"test_class_{int(label)}_data"))
                    if gen_index > 0:
                        ro_index = [np.random.choice(gen_index)]
                    else:
                        ro_index = [np.random.choice(gen_index+1)]
                    #ro_index = [np.random.choice(len(getattr(self.args.dataset,f"test_class_{rand_label}_data")) % len(getattr(self.args.dataset, f"test_class_{int(label)}_data")))]
                    Ro, ro_weight = [getattr(self.args.dataset, f"test_class_{min_class}_data")[ro_index[0]]], [1.0]
                elif self.args.grad_replacement == 'max_class_random_instance':
                    gen_index = len(getattr(self.args.dataset, f"test_class_{max_class}_data")) % len(
                        getattr(self.args.dataset, f"test_class_{int(label)}_data"))
                    if gen_index > 0:
                        ro_index = [np.random.choice(gen_index)]
                    else:
                        ro_index = [np.random.choice(gen_index+1)]
                    #ro_index = [np.random.choice(len(getattr(self.args.dataset,f"test_class_{rand_label}_data")) % len(getattr(self.args.dataset, f"test_class_{int(label)}_data")))]
                    Ro, ro_weight = [getattr(self.args.dataset, f"test_class_{max_class}_data")[ro_index[0]]], [1.0]
                elif self.args.grad_replacement == 'next_class_random_instance':

                    if self.args.first_class == 'yes':
                        next_class = category_one
                    elif self.args.second_class == 'yes':
                        next_class = category_two
                    # elif self.args.third_class == 'yes':
                    #     next_class = category_three
                    else:
                        next_class = min_class

                    gen_index = len(getattr(self.args.dataset, f"test_class_{next_class}_data")) % len(
                        getattr(self.args.dataset, f"test_class_{int(label)}_data"))
                    if gen_index > 0:
                        ro_index = [np.random.choice(gen_index)]
                    else:
                        ro_index = [np.random.choice(gen_index+1)]
                    #ro_index = [np.random.choice(len(getattr(self.args.dataset,f"test_class_{rand_label}_data")) % len(getattr(self.args.dataset, f"test_class_{int(label)}_data")))]
                    Ro, ro_weight = [getattr(self.args.dataset, f"test_class_{next_class}_data")[ro_index[0]]], [1.0]
                elif self.args.grad_replacement == 'min_class_mean':
                    ro_index = 0
                    Ro, ro_weight = [getattr(self.args.dataset, f"test_class_{next_class}_mean")], [1.0]
                else:
                    ro_index = [np.random.choice(len(getattr(self.args.dataset, f"test_class_{1-int(label)}_data")))]
                    Ro, ro_weight = [getattr(self.args.dataset, f"test_class_{1-int(label)}_data")[ro_index[0]]], [1.0]

            else:
                ro_index = [np.random.choice(len(getattr(self.args.dataset, f"test_class_{1-int(label)}_data")))]
                Ro, ro_weight = [getattr(self.args.dataset, f"test_class_{1-int(label)}_data")[ro_index[0]]], [1.0]

        else:
            self.mode = 'Exploit'
            Rs, rs_weight, rs_index = self.rs_priority_buffer.sample(1)
            Ro, ro_weight, ro_index = self.ro_priority_buffer.sample(1)
        return {'rs': [Rs, rs_weight, rs_index],
                'ro': [Ro, ro_weight, ro_index]}

    def dynamic_dual_pick_zt(self, kwargs, data, label):
        ds = kwargs['dataset'].test_class_0_indices
        ls = len(ds)
        ods = kwargs['dataset'].test_class_1_indices
        ols = len(ods)
        self.r_index = random.randrange(0, ls)
        self.ro_index = random.randrange(0, ols)
        Zt = self.background_data[ds[self.r_index]]
        ZOt = self.background_data[ods[self.ro_index]]
        return Zt, ZOt

    def dynamic_pick_zt(self, kwargs, data, label):
        self.r_index = None
        if self.args.grad_replacement == 'zeros':
            Zt = torch.zeros_like(data)
        else:
            if self.args.grad_replacement == 'class_mean':
                if label == 1:
                    Zt = torch.tensor(kwargs['dataset'].test_class_0_mean, dtype=torch.float32)
                else:
                    Zt = torch.tensor(kwargs['dataset'].test_class_1_mean, dtype=torch.float32)
            elif self.args.grad_replacement == 'instance_mean':
                Zt = torch.mean(data).cpu().detach().numpy()
                Zt = torch.tensor(np.repeat(Zt, data.shape[0]), dtype=torch.float32)
            elif self.args.grad_replacement == 'random_instance':
                self.r_index = random.randrange(0, len(self.background_data))
                Zt = torch.tensor(self.background_data[self.r_index],
                                  dtype=torch.float32)
            elif self.args.grad_replacement == 'random_opposing_instance':
                if label == 1:
                    sds = kwargs['dataset'].test_class_0_indices
                    sls = len(sds)
                else:
                    sds = kwargs['dataset'].test_class_1_indices
                    sls = len(sds)
                self.r_index = random.randrange(0, sls)
                Zt = torch.tensor(sds[self.r_index], dtype=torch.float32)
        return Zt

    def static_pick_zt(self, kwargs, data, label):
        if self.args.grad_replacement == 'zeros':
            Zt = torch.zeros_like(data)
        else:
            if self.args.grad_replacement == 'class_mean':
                if label == 1:
                    Zt = torch.tensor(kwargs['dataset'].test_class_0_mean, dtype=torch.float32)
                else:
                    Zt = torch.tensor(kwargs['dataset'].test_class_1_mean, dtype=torch.float32)
            elif self.args.grad_replacement == 'instance_mean':
                Zt = torch.mean(data).cpu().detach().numpy()
                Zt = torch.tensor(np.repeat(Zt, data.shape[0]), dtype=torch.float32)
            elif self.args.grad_replacement == 'random_instance':
                Zt = torch.tensor(self.background_data[self.r_index], dtype=torch.float32)
            elif self.args.grad_replacement == 'random_opposing_instance':
                if label == 1:
                    Zt = torch.tensor(kwargs['dataset'].test_statistics['between_class']['opposing'][0],
                                      dtype=torch.float32)
                else:
                    Zt = torch.tensor(kwargs['dataset'].test_statistics['between_class']['opposing'][1],
                                      dtype=torch.float32)
        return Zt

    def weighted_mse_loss(self, input, target, weight):
        return torch.mean(weight * (input - target) ** 2)

    def euc(self, point, dist, cov=None):
        return scipy.spatial.distance.euclidean(point, np.mean(dist, axis=0))
        # print(point.shape, dist.shape)
        # print(dist.sum(axis=1))
        # # exit()
        # point = (point-np.min(point))/(np.max(point)-np.min(point)+ 1e-27)
        # dist = np.array([ ((dist[d]-np.min(dist[d]))/(np.max(dist[d]) - np.min(dist[d]) + 1e-27)) for d in range(len(dist))])


    def mahalanobis(self, point, dist, cov=None):
        # return scipy.spatial.distance.euclidean(point, np.mean(dist, axis=0))
        # print(point.shape, dist.shape)
        # print(dist.sum(axis=1))
        # # exit()
        # point = (point-np.min(point))/(np.max(point)-np.min(point)+ 1e-27)
        # dist = np.array([ ((dist[d]-np.min(dist[d]))/(np.max(dist[d]) - np.min(dist[d]) + 1e-27)) for d in range(len(dist))])
        try:
            x_mean = np.mean(point)
            # print(x_mean)
            Covariance = np.cov(np.transpose(dist))
            # Covariance = np.cov((dist))
            # print(Covariance)
            # inv_covmat = np.linalg.inv(Covariance)
            inv_covmat = np.linalg.pinv(Covariance)
            x_minus_mn = point - x_mean
            # print(x_mean, Covariance, inv_covmat)
            D_square = np.dot(np.dot(x_minus_mn, inv_covmat), np.transpose(x_minus_mn))
            # print(D_square.diagonal())
            return D_square.diagonal()
            # return (D_square.diagonal()*D_square.diagonal())
        except Exception as e:
            # print(e)
            # exit()
            return "0.0"

    def generate_saliency(self, data, label, **kwargs):

        if isinstance(data, np.ndarray):
            # data = numpy_to_torch(data, use_cuda=False, requires_grad=False)
            data = torch.tensor(data.flatten(), dtype=torch.float32)

        orig_label = label
        flag = 0
        orig_pred_prob = (kwargs['target'].cpu().data.numpy())
        category = np.argmax(kwargs['target'].cpu().data.numpy())
        orig_sorted_category = np.argsort(kwargs['target'].cpu().data.numpy())
        ordered_orig_sorted_category = np.argsort(kwargs['target'].cpu().data.numpy()*-1)
        num_classes = getattr(kwargs['dataset'], f"num_classes")

        categories = ordered_orig_sorted_category.tolist()
        max_category = categories[0]

        label_category_pred_prob = kwargs['target'].cpu().data.numpy()[label]

        per_class_masks = []
        for top_n in range(num_classes):

            top_class = top_n
            c = top_n

            # Initialize rs_priority_buffer
            self.rs_priority_buffer = PrioritizedBuffer(background_data=getattr(kwargs['dataset'], f"test_class_{int(top_class)}_data"))

            class_choices = list(range(0, int(top_class))) + list(range(int(top_class)+1, num_classes))
            if self.args.p_buf_init == "random":
                rand_label = np.random.choice(class_choices)
                self.ro_priority_buffer = PrioritizedBuffer(background_data=getattr(
                    kwargs['dataset'], f"test_class_{rand_label}_data"))
            elif self.args.p_buf_init == "maha":
                dist = [self.mahalanobis(data.numpy().reshape(1, -1), getattr(
                    kwargs['dataset'], f"test_class_{c}_data")) for c in class_choices]
                self.ro_priority_buffer = PrioritizedBuffer(background_data=getattr(
                    kwargs['dataset'], f"test_class_{np.argmax(dist)}_data"))
            elif self.args.p_buf_init == "euc":
                dist = [self.euc(data.numpy().reshape(1, -1), getattr(
                    kwargs['dataset'], f"test_class_{c}_data")) for c in class_choices]
                self.ro_priority_buffer = PrioritizedBuffer(background_data=getattr(
                    kwargs['dataset'], f"test_class_{np.argmax(dist)}_data"))
            elif self.args.p_buf_init == "cluster":
                centroids = getattr(kwargs['dataset'], f"cluster_meta")["centroids"]
                dist = [scipy.spatial.distance.euclidean(
                    data.numpy().reshape(1, -1), centroids[c]) for c in range(getattr(kwargs['dataset'], f"cluster_meta")["n_clusters"])]
                indices = getattr(kwargs['dataset'], "cluster_meta")[f"cluster_{np.argmax(dist)}_indices"]
                self.ro_priority_buffer = PrioritizedBuffer(background_data=np.take(getattr(kwargs['dataset'], "train_data"), indices[0], axis=0))

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
            #TODO for args.second_class, args.third_class onwards mask_init should be proportional or normalized
            mask_init = np.full(shape=len(data), fill_value=1e-2)
            blurred_signal_norm = blurred_signal / np.max(blurred_signal)
            plt.plot(blurred_signal_norm, label="Blur Norm")

            if self.enable_wandb:
                wandb.log({'Initialization': plt}, step=0)

            blurred_signal = torch.tensor((blurred_signal_norm.reshape(1, -1)), dtype=torch.float32)
            # mask = numpy_to_torch(mask_init, use_cuda=self.use_cuda)
            mask_init = np.random.uniform(size=len(data), low=-1e-2, high=1e-2)
            n_mask = numpy_to_torch(mask_init, use_cuda=self.use_cuda)

            optimizer = torch.optim.Adam([n_mask], lr=self.args.lr)

            if self.args.enable_lr_decay:
                scheduler = ExponentialLR(optimizer, gamma=self.args.lr_decay)

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

            hinge_loss_fn = nn.HingeEmbeddingLoss()
            triplet_loss_fn = nn.TripletMarginLoss(margin=1.0, p=2)
            cosine_loss_fn = nn.CosineSimilarity(dim=1, eps=1e-6)
            mse_loss_fn = torch.nn.MSELoss(reduction='mean')

            mse_loss_fn_only = torch.nn.MSELoss(reduction='mean')

            softdtw_loss_fn = SoftDTW()
            original_class_predictions = torch.max(kwargs['target']).item()

            for i in (range(self.args.max_itr)):

                CUR_DIR = kwargs['save_dir'] + f'/epoch_{i}/'

                picks = self.priority_dual_greedy_pick_rt(kwargs=kwargs, data=data, label=top_class)
                # picks = self.priority_dual_greedy_pick_rt(kwargs=kwargs, data=data, label=label)
                rs, rs_weight, rs_index = picks['rs']
                ro, ro_weight, ro_index = picks['ro']
                Rt = []

                for e, (rs_e, ro_e, m) in enumerate(zip(rs[0].flatten(), ro[0].flatten(), n_mask.detach().numpy().flatten())):
                    if m < 0:
                        Rt.append(rs_e)
                    else:
                        Rt.append(ro_e)
                Rt = torch.tensor(Rt, dtype=torch.float32)

                perturbated_input = data.mul(n_mask) + Rt.mul(1 - n_mask)

                if self.args.enable_blur:
                    perturbated_input += blurred_signal.mul(n_mask)

                if self.args.enable_noise:
                    noise = np.zeros(data.shape, dtype=np.float32)
                    cv2.randn(noise, 0, 0.3)
                    noise = numpy_to_torch(noise, use_cuda=self.use_cuda)
                    perturbated_input += noise

                perturbated_input = perturbated_input.flatten()

                outputs = self.softmax_fn(self.predict_fn(perturbated_input.reshape([1, -1])))[0]
                metrics['Confidence'].append(float(outputs[top_class].item()))
                if kwargs['save_perturbations'] and not self.args.run_eval_every_epoch:
                    self.perturbation_manager.add_perturbation(perturbation=perturbated_input.cpu(
                    ).detach().numpy().flatten(), step=i, confidence=metrics['Confidence'][-1])

                c_mse = self.args.mse_coeff * mse_loss_fn(outputs[top_class], kwargs['target'][top_class])

                c_budget = self.args.l1_coeff * torch.mean(torch.abs(n_mask)) * float(self.args.enable_budget)

                c_tvnorm = self.args.tv_coeff * tv_norm(n_mask, self.args.tv_beta) * float(self.args.enable_tvnorm)

                mse_var = np.var(metrics['MSE']) if len(metrics['MSE']) > 1 else 0.0

                dist_loss = torch.tensor(0.0)

                loss = c_mse + c_budget + c_tvnorm

                if loss < 0:
                    rs_prios = loss * (rs_weight[0])*0.0
                    ro_prios = loss * (ro_weight[0])*0.0
                else:
                    rs_prios = loss * (rs_weight[0])
                    ro_prios = loss * (ro_weight[0])

                loss = loss * (rs_weight[0] + ro_weight[0]) / 2

                optimizer.zero_grad()
                loss.backward()

                metrics['Mean Gradient'].append(float(np.mean(n_mask.grad.cpu().detach().numpy())))
                metrics['TV Norm'].append(float(c_tvnorm.item()))
                metrics['MSE'].append(float(c_mse.item()))
                metrics['Budget'].append(float(c_budget.item()))
                metrics['Total Loss'].append(float(loss))
                metrics['Category'].append(int(np.argmax(outputs.cpu().detach().numpy())))
                metrics['Saliency Sum'].append(float(np.sum(n_mask.cpu().detach().numpy() > 0)))
                metrics['Saliency Var'].append(float(np.var(n_mask.cpu().detach().numpy())))
                metrics['MSE Coeff'].append(self.args.mse_coeff)
                metrics['TV Coeff'].append(self.args.tv_coeff)
                metrics['L1 Coeff'].append(self.args.l1_coeff)
                metrics['MSE Var'].append(mse_var)

                torch.set_printoptions(precision=2)

                optimizer.step()

                self.rs_priority_buffer.update_priorities(rs_index, [rs_prios.item()])
                self.ro_priority_buffer.update_priorities(ro_index, [ro_prios.item()])

                if self.args.enable_lr_decay:
                    scheduler.step(epoch=i)

                n_mask.data.clamp_(-1, 1)

                if self.args.run_eval_every_epoch:
                    m = n_mask.cpu().detach().numpy().flatten()
                    metrics["epoch"][f"epoch_{i}"] = {'eval_metrics': run_evaluation_metrics(
                        self.args.eval_replacement, kwargs['dataset'], data, self.predict_fn, m, kwargs['save_dir'], False)}

                    if kwargs['save_perturbations']:
                        self.perturbation_manager.add_perturbation(
                            perturbation=perturbated_input.cpu().detach().numpy().flatten(),
                            saliency=n_mask.cpu().detach().numpy(),
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

                    wandb.log(_mets)
            
            per_class_masks.append(n_mask.data.squeeze(0).squeeze(0).numpy())

        main_saliency = per_class_masks[max_category]

        np.set_printoptions(suppress=True)
        np.set_printoptions(precision=3)
        main_saliency = main_saliency.astype(np.float32)
        upsampled_mask = torch.tensor(main_saliency, dtype=torch.float32)
        mask = torch.tensor(main_saliency, dtype=torch.float32)

        np.save(kwargs['save_dir'] + "/upsampled_mask", upsampled_mask.cpu().detach().numpy())
        np.save(kwargs['save_dir'] + "/mask", mask.cpu().detach().numpy())
        if self.enable_wandb:
            wandb.save(kwargs['save_dir'] + "/metrics.json")
            wandb.save(kwargs['save_dir'] + "/upsampled_mask.npy")
            wandb.save(kwargs['save_dir'] + "/mask.npy")

        # mask.data.clamp_(0,1)
        mask = mask.squeeze(0).squeeze(0)
        mask = mask.cpu().detach().numpy().flatten()

        if self.enable_wandb:
            wandb.run.summary["pos_features"] = len(np.where(mask > 0)[0])
            wandb.run.summary["neg_features"] = len(np.where(mask < 0)[0])
            wandb.run.summary["pos_sum"] = np.sum(mask[np.argwhere(mask > 0)])
            wandb.run.summary["neg_sum"] = np.sum(mask[np.argwhere(mask < 0)])

        norm_mask = mask
        save_timeseries(mask=norm_mask, raw_mask=mask, time_series=data.numpy(), blurred=blurred_signal,
                        save_dir=kwargs['save_dir'], enable_wandb=self.enable_wandb, algo=self.args.algo, dataset=self.args.dataset, category=top_class)

        if self.enable_wandb:
            wandb.run.summary["norm_saliency_sum"] = np.sum(mask)
            wandb.run.summary["norm_saliency_var"] = np.var(mask)
            wandb.run.summary["norm_pos_sum"] = np.sum(norm_mask[np.argwhere(mask > 0)])
            wandb.run.summary["norm_neg_sum"] = np.sum(norm_mask[np.argwhere(mask < 0)])
            wandb.run.summary["label"] = top_class

        np.save("./r_mask", mask)
        np.save("./n_mask", norm_mask)

        top_label = max_category
        top_class_mask = per_class_masks[top_label]
        distinct_mask = 0
        for c in range(num_classes):
            if c != top_label:
                distinct_mask += (top_class_mask - per_class_masks[c])
        dmin, dmax = np.min(distinct_mask), np.max(distinct_mask)
        den = dmax-dmin
        if den == 0:
            den = 1e-27
        distinct_mask = 2 * ((distinct_mask-dmin)/(den)) - 1
        saliency = distinct_mask
        return saliency, self.perturbation_manager, np.array(per_class_masks)
