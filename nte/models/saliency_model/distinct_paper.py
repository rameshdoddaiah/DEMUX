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
from sklearn.manifold import TSNE

  
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



class distinct_paper_saliency(Saliency):
    def __init__(self, background_data, background_label, predict_fn, enable_wandb, use_cuda, args):
        super(distinct_paper_saliency, self).__init__(background_data=background_data,background_label=background_label,predict_fn=predict_fn)
        self.enable_wandb = enable_wandb
        self.use_cuda = use_cuda
        self.args = args
        self.softmax_fn = torch.nn.Softmax(dim=-1)
        self.perturbation_manager = None
        self.r_index = random.randrange(0, len(self.background_data)) if self.args.r_index < 0 else self.args.r_index
        self.rs_priority_buffer = None
        self.ro_priority_buffer = None
        self.eps = 1.0
        self.eps_decay = 0.9991#0.9996#0.997


    def priority_dual_greedy_pick_rt(self, kwargs, data, label):
        self.eps *= self.eps_decay
        if np.random.uniform() < self.eps:
            self.mode = 'Explore'
            # todo: Change Rs and Ro dynamically
            num_classes = getattr(kwargs['dataset'],f"num_classes")
            rs_index = [np.random.choice(len(getattr(self.args.dataset, f"test_class_{int(label)}_data")))]
            Rs, rs_weight = [getattr(self.args.dataset, f"test_class_{int(label)}_data")[rs_index[0]]], [1.0]

            if num_classes >2:

                min_class = np.argmin(kwargs['target'].cpu().data.numpy())
                max_class = np.argmax(kwargs['target'].cpu().data.numpy())
                orig_sorted_category = np.argsort(kwargs['target'].cpu().data.numpy())
                category_zero = orig_sorted_category[-1]
                category_one = orig_sorted_category[-2]
                category_two = orig_sorted_category[-3]

                if self.args.grad_replacement == 'random_instance':
                    rand_choice = list(range(0,int(label))) + list(range(int(label)+1, num_classes))
                    rand_label = np.random.choice(rand_choice)
                    gen_index =  len(getattr(self.args.dataset,f"test_class_{rand_label}_data")) % len(getattr(self.args.dataset, f"test_class_{int(label)}_data"))
                    if gen_index >0:
                        ro_index = [np.random.choice(gen_index)]
                    else:
                        ro_index = [np.random.choice(gen_index+1)]
                    #ro_index = [np.random.choice(len(getattr(self.args.dataset,f"test_class_{rand_label}_data")) % len(getattr(self.args.dataset, f"test_class_{int(label)}_data")))]
                    Ro, ro_weight = [getattr(self.args.dataset, f"test_class_{rand_label}_data")[ro_index[0]]], [1.0]
                elif self.args.grad_replacement == 'min_class_random_instance':
                    gen_index =  len(getattr(self.args.dataset,f"test_class_{min_class}_data")) % len(getattr(self.args.dataset, f"test_class_{int(label)}_data"))
                    if gen_index >0:
                        ro_index = [np.random.choice(gen_index)]
                    else:
                        ro_index = [np.random.choice(gen_index+1)]
                    #ro_index = [np.random.choice(len(getattr(self.args.dataset,f"test_class_{rand_label}_data")) % len(getattr(self.args.dataset, f"test_class_{int(label)}_data")))]
                    Ro, ro_weight = [getattr(self.args.dataset, f"test_class_{min_class}_data")[ro_index[0]]], [1.0]
                elif self.args.grad_replacement == 'max_class_random_instance':
                    gen_index =  len(getattr(self.args.dataset,f"test_class_{max_class}_data")) % len(getattr(self.args.dataset, f"test_class_{int(label)}_data"))
                    if gen_index >0:
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

                    gen_index =  len(getattr(self.args.dataset,f"test_class_{next_class}_data")) % len(getattr(self.args.dataset, f"test_class_{int(label)}_data"))
                    if gen_index >0:
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
            print("Mahalanobis exception")
            return "0.0"

    def generate_saliency(self, data, label, **kwargs):

        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=torch.float32)

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

        if (self.args.top_n == 'yes'):
            # top_n_range = num_classes
            top_n_range = 1
        else:
            top_n_range = 1

        for top_n in range(top_n_range):

            various_perturbations = []
            various_perturbations_labels = []
            various_perturbations_masks = []
            reached_perturbs = 0

            # Initialize rs_priority_buffer
            self.rs_priority_buffer = PrioritizedBuffer(background_data=getattr(
                kwargs['dataset'], f"test_class_{int(label)}_data"))

            # Initialize ro_priority_buffer
            if num_classes > 2:
                class_choices = list(range(0, int(label))) + list(range(int(label)+1, num_classes))
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
                    self.ro_priority_buffer = PrioritizedBuffer(background_data=np.take(getattr(kwargs['dataset'], "train_data"), indices, axis=0))
            else:
                self.ro_priority_buffer = PrioritizedBuffer(background_data=getattr(
                    kwargs['dataset'], f"test_class_{1 - int(label)}_data"))

            self.eps = 1.0

            if kwargs['save_perturbations']:
                self.perturbation_manager = PerturbationManager(original_signal=data.cpu().detach().numpy().flatten(),algo=self.args.algo, prediction_prob=np.max(kwargs['target'].cpu().data.numpy()),
                    original_label=label, sample_id=self.args.single_sample_id)

            plt.plot(data, label="Original Signal Norm")
            gkernel = cv2.getGaussianKernel(3, 0.5)
            gaussian_blur_signal = cv2.filter2D(data.cpu().detach().numpy(), -1, gkernel).flatten()
            plt.plot(gaussian_blur_signal, label="Gaussian Blur")
            median_blur_signal = median_filter(data, 3)
            plt.plot(median_blur_signal, label="Median Blur")
            blurred_signal = (gaussian_blur_signal + median_blur_signal) / 2
            plt.plot(blurred_signal, label="Blurred Signal")
            mask_init = np.random.uniform(size=len(data), low=-1e-2, high=1e-2)
            #TODO for args.second_class, args.third_class onwards mask_init should be proportional or normalized
            # mask_init = np.full(shape=len(data), fill_value=1e-2)
            blurred_signal_norm = blurred_signal / np.max(blurred_signal)
            plt.plot(blurred_signal_norm, label="Blur Norm")

            if self.enable_wandb:
                wandb.log({'Initialization': plt}, step=0)


            blurred_signal = torch.tensor((blurred_signal_norm.reshape(1, -1)), dtype=torch.float32)
            n_mask = []
            # mask = numpy_to_torch(mask_init, use_cuda=self.use_cuda)
            for i in range(num_classes):
                mask_init = np.random.uniform(size=len(data), low=-1e-2, high=1e-2)
                n_mask.append(numpy_to_torch(mask_init, use_cuda=self.use_cuda))

            optimizer = torch.optim.Adam([n_mask[c] for c in range(num_classes)], lr=self.args.lr)

            if self.args.enable_lr_decay:
                scheduler = ExponentialLR(optimizer, gamma=self.args.lr_decay)

            print(f"{self.args.algo}: Category with highest probability {category}")
            print(f"{self.args.algo}: Optimizing.. ")

            metrics = {"TV Norm": [], 'TV Coeff': [], "MSE": [], "Budget": [], "Total Loss": [],
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
            if (self.args.ce == 'yes'):
                mse_loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
            elif(self.args.kl == 'yes'):
                mse_loss_fn = torch.nn.functional.kl_div
            elif(self.args.mse == 'yes'):
                mse_loss_fn = torch.nn.MSELoss(reduction='mean')
            else:
                mse_loss_fn = torch.nn.MSELoss(reduction='mean')

            softdtw_loss_fn = SoftDTW()
            original_class_predictions = torch.max(kwargs['target']).item()

            for i in (range(self.args.max_itr)):

                CUR_DIR = kwargs['save_dir'] + f'/epoch_{i}/'

                picks = self.priority_dual_greedy_pick_rt(kwargs=kwargs, data=data, label=label)
                rs, rs_weight, rs_index = picks['rs']
                ro, ro_weight, ro_index = picks['ro']
                Rt = [[0]*data.shape[0]] * num_classes

                for c in range(num_classes):
                    for e, (rs_e, ro_e, m) in enumerate(zip(rs[0].flatten(), ro[0].flatten(), n_mask[c].detach().numpy().flatten())):
                        if m < 0:
                            Rt[c][e] = rs_e
                            # Rm.append("z")
                        else:
                            Rt[c][e] = ro_e
                            # Rm.append("o")
                    Rt[c] = torch.tensor(Rt[c], dtype=torch.float32)

                perturbated_input = []

                for c in range(num_classes):
                    perturbated_input.append(data.mul(n_mask[c]) + Rt[c].mul(1 - n_mask[c]))

                if self.args.enable_blur:
                    for c in range(num_classes):
                        perturbated_input[c] += blurred_signal.mul(n_mask[c])

                if self.args.enable_noise:
                    noise = np.zeros(data.shape, dtype=np.float32)
                    cv2.randn(noise, 0, 0.3)
                    noise = numpy_to_torch(noise, use_cuda=self.use_cuda)
                    for c in range(num_classes):
                        perturbated_input[c] += noise

                for c in range(num_classes):
                    perturbated_input[c] = perturbated_input[c].flatten()

                outputs = [[0]*num_classes]*num_classes
                if (self.args.mse == 'yes'):
                    for c in range(num_classes):
                        outputs[c] = self.softmax_fn(self.predict_fn(perturbated_input[c].reshape([1, -1])))[0]
                    metrics['Confidence'].append(float(outputs[top_class][label].item()))
                    metrics['Top Confidence'].append(float(outputs[top_class][top_class].item()))

                if kwargs['save_perturbations'] and not self.args.run_eval_every_epoch:
                    self.perturbation_manager.add_perturbation(perturbation=perturbated_input.cpu().detach().numpy().flatten(), step=i, confidence=metrics['Confidence'][-1])
                if self.args.bbm == 'rnn':
                    c0 = self.args.mse_coeff * mse_loss_fn(outputs[category_zero], kwargs['target'].squeeze(0)[category_zero])
                else:
                    c_mse = []
                    if(self.args.mse == 'yes'):
                        for c in range(num_classes):
                            # c_mse.append(self.args.mse_coeff * mse_loss_fn(outputs[c][categories[c]], kwargs['target'][categories[c]]))
                            # c_mse.append(self.args.mse_coeff * mse_loss_fn(outputs[c][c], kwargs['target'][c]))
                            c_mse.append(self.args.mse_coeff * mse_loss_fn(outputs[c], kwargs['target']))

                c_budget = []
                for c in range(num_classes):
                    c_budget.append(self.args.l1_coeff * torch.mean(torch.abs(n_mask[c])) * float(self.args.enable_budget))

                c_tvnorm = []
                for c in range(num_classes):
                    c_tvnorm.append(self.args.tv_coeff * tv_norm(n_mask[c], self.args.tv_beta) * float(self.args.enable_tvnorm))

                mask_mse_loss = []
                mse_var = np.var(metrics['MSE']) if len(metrics['MSE']) > 1 else 0.0

                dist_loss = torch.tensor(0.0)

                loss = 0
                for c in range(num_classes):
                    loss = loss + c_mse[c] + c_budget[c] + c_tvnorm[c]
                        
                loss = loss + dist_loss

                if loss < 0:
                    rs_prios = loss * (rs_weight[0])*0.0
                    ro_prios = loss * (ro_weight[0])*0.0
                else:
                    rs_prios = loss * (rs_weight[0])
                    ro_prios = loss * (ro_weight[0])

                loss = loss * (rs_weight[0] + ro_weight[0]) / 2

                abs_mask_t = n_mask[top_class].detach().clone().numpy()
                mask_min_t= np.min(abs_mask_t)
                mask_max_t = np.max(abs_mask_t)

                delta_t = mask_max_t-mask_min_t
                if (delta_t) == 0:
                    add_noise = 1e-27
                else:
                    add_noise = 0
                norm_mask_t = (abs_mask_t - mask_min_t) / (mask_max_t - mask_min_t + add_noise)

                optimizer.zero_grad()
                loss.backward()
                metrics['Mean Gradient'].append(float(np.mean(n_mask[top_class].grad.cpu().detach().numpy())))
                metrics['TV Norm'].append(float(c_tvnorm[top_class].item()))
                metrics['MSE'].append(float(c_mse[top_class].item()))
                metrics['DIST'].append(float(dist_loss.item()))
                metrics['Budget'].append(float(c_budget[top_class].item()))
                metrics['Total Loss'].append(float(loss))
                metrics['Category'].append(int(np.argmax(outputs[top_class].cpu().detach().numpy())))
                metrics['Saliency Sum'].append(float(np.sum(n_mask[top_class].cpu().detach().numpy()>0)))
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
                        f"| Top_class : {top_class} Reached Prob:{outputs[top_class][label]:.4f}"
                        f"| Prob : {kwargs['target'].cpu().data.numpy()[top_class]:.4f}"
                    )

                optimizer.step()

                self.rs_priority_buffer.update_priorities(rs_index, [rs_prios.item()])
                self.ro_priority_buffer.update_priorities(ro_index, [ro_prios.item()])

                if self.args.enable_lr_decay:
                    scheduler.step(epoch=i)

                for c in range(num_classes):
                    n_mask[c].data.clamp_(-1,1)

                if self.args.run_eval_every_epoch:
                    m = n_mask[top_class].cpu().detach().numpy().flatten()
                    metrics["epoch"][f"epoch_{i}"] = {'eval_metrics': run_evaluation_metrics(self.args.eval_replacement, kwargs['dataset'], data, self.predict_fn, m, kwargs['save_dir'], False)}

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

                if self.enable_wandb:
                    _mets = {**{k: v[-1] for k, v in metrics.items() if k != "epoch"},
                            **{"Gradient": wandb.Histogram(n_mask[top_class].grad.cpu().detach().numpy()),
                                "Training": [n_mask[top_class], noise, perturbated_input,
                                            ] + [perturbation] if self.args.enable_blur else []
                                }
                            }
                    if f"epoch_{i}" in metrics["epoch"]:
                        _mets = {**_mets, **metrics["epoch"]
                                [f"epoch_{i}"]['eval_metrics']}

                    wandb.log(_mets)

                CRED = '\033[91m'
                CEND = '\033[0m'
                CGRN = '\033[92m'
                CGRN = '\033[92m'
                CCYN = "\033[36m"

        for mse_c in range(num_classes):
          if self.args.top_class == 'yes':
              if mse_c != top_class:
                  continue
          top_class_mask = n_mask[mse_c].clone()
          distinct_mask_t = torch.zeros_like(n_mask[mse_c])
          top_class_mask.data.clamp_(0, 1)
          for c in range(num_classes):
            #   class_prob = kwargs['target'].cpu().data.numpy()[mse_c]
              class_prob = 1
              if (c != mse_c) and (c == ordered_orig_sorted_category[0] or c == ordered_orig_sorted_category[1] or c == ordered_orig_sorted_category[2]):
                  mask_t2 = n_mask[c].clone()
                  mask_t2.data.clamp_(0, 1)
                #   np.maximum(array,0)
                  max_mask_t3 = np.maximum(top_class_mask.data - mask_t2.data,0)
                #   max_mask_t3 = np.maximum(top_class_mask.data - mask_t2.data*self.args.class_prob,0)
                #   max_mask_t3.data.clamp_(0, 1)
                  distinct_mask_t = distinct_mask_t + max_mask_t3

        #we need to normalize. use min and max normalization instead of clamping
        # All classes Distinct
        # only 3 class 
        # Confidence > 0.1 or 0.2 until 0.9
        # LMUX
        # All classes
        # only 3 class
        dmin = np.min(np.array(distinct_mask_t))
        dmax = np.max(np.array(distinct_mask_t))
        distinct_mask_t = (distinct_mask_t-dmin)/(dmax-dmin + 1e-27)
        # distinct_mask_t.data.clamp_(0, 1)
        various_perturbations_masks_mean = distinct_mask_t.data.numpy()
        mask_t = various_perturbations_masks_mean
        main_saliency = various_perturbations_masks_mean

        np.set_printoptions(suppress=True)
        np.set_printoptions(precision=3)
        main_saliency = main_saliency
        main_saliency = main_saliency.astype(np.float32)
        if self.args.bbm == 'rnn':
            if self.args.bbmg == 'yes':
                timesteps = int(perturbated_input.shape[0]/self.args.window_size)
                Final_unique_prob_pred = self.softmax_fn(self.predict_fn((data*main_saliency).reshape(-1,timesteps,self.args.window_size))).unsqueeze(0)
            else:
                Final_unique_prob_pred = self.softmax_fn(self.predict_fn((data*main_saliency).reshape(1,-1))).unsqueeze(0)
        else:
            Final_unique_prob_pred = self.softmax_fn(self.predict_fn(data*main_saliency))
        various_perturbs_pred = self.softmax_fn(self.predict_fn(torch.tensor(main_saliency,dtype=torch.float32).reshape([1, -1])))
        outputs = (self.predict_fn(perturbated_input[top_class].reshape([1, -1])))
        Final_diff = Final_unique_prob_pred - orig_pred_prob
        delta_pred_prob = Final_unique_prob_pred[0][0][top_class] - kwargs['target'].cpu().data.numpy()[top_class]
        upsampled_mask = torch.tensor(main_saliency,dtype=torch.float32)
        mask = torch.tensor(main_saliency,dtype=torch.float32)
        np.save(kwargs['save_dir'] + "/upsampled_mask", upsampled_mask.cpu().detach().numpy())
        np.save(kwargs['save_dir'] + "/mask", mask.cpu().detach().numpy())
        if self.enable_wandb:
            wandb.save(kwargs['save_dir'] + "/metrics.json")
            wandb.save(kwargs['save_dir'] + "/upsampled_mask.npy")
            wandb.save(kwargs['save_dir'] + "/mask.npy")

        mask = mask.squeeze(0).squeeze(0)
        mask = mask.cpu().detach().numpy().flatten()

        if self.enable_wandb:
            wandb.run.summary["prediction_probability_diff"] = delta_pred_prob
            wandb.run.summary["pos_features"] = len(np.where(mask > 0)[0])
            wandb.run.summary["neg_features"] = len(np.where(mask < 0)[0])
            wandb.run.summary["pos_sum"] = np.sum(mask[np.argwhere(mask > 0)])
            wandb.run.summary["neg_sum"] = np.sum(mask[np.argwhere(mask < 0)])

        norm_mask  = mask

        if self.args.bbm == 'rnn':
            if self.args.bbmg == 'yes':
                timesteps = int(perturbated_input.shape[0]/self.args.window_size)
                Final_unique_prob_pred = self.softmax_fn(self.predict_fn((data*norm_mask).reshape(-1,timesteps,self.args.window_size))).unsqueeze(0)
            else:
                Final_unique_prob_pred = self.softmax_fn(self.predict_fn((data*norm_mask).reshape(1,-1))).unsqueeze(0)
        else:
            Final_unique_prob_pred = self.softmax_fn(self.predict_fn(data*norm_mask))

        save_timeseries(mask=norm_mask, raw_mask=mask, time_series=data.numpy(),blurred=blurred_signal, save_dir=kwargs['save_dir'], enable_wandb=self.enable_wandb,algo=self.args.algo, dataset=self.args.dataset, category=label)

        if self.enable_wandb:
            wandb.run.summary["norm_saliency_sum"] = np.sum(mask)
            wandb.run.summary["norm_saliency_var"] = np.var(mask)
            wandb.run.summary["norm_pos_sum"] = np.sum(norm_mask[np.argwhere(mask > 0)])
            wandb.run.summary["norm_neg_sum"] = np.sum(norm_mask[np.argwhere(mask < 0)])
            wandb.run.summary["label"] = label

        np.save("./r_mask", mask)
        np.save("./n_mask", norm_mask)

        return mask, self.perturbation_manager
