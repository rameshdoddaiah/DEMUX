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


class NTEGradientSaliency(Saliency):
    def __init__(self, background_data, background_label, predict_fn, enable_wandb, use_cuda, args):
        super(NTEGradientSaliency, self).__init__(background_data=background_data, background_label=background_label,
                                                  predict_fn=predict_fn)
        self.enable_wandb = enable_wandb
        self.use_cuda = use_cuda
        self.args = args
        self.softmax_fn = torch.nn.Softmax(dim=-1)
        self.perturbation_manager = None
        self.r_index = random.randrange(0, len(self.background_data)) if self.args.r_index < 0 else self.args.r_index

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

    def generate_saliency(self, data, label, **kwargs):

        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=torch.float32)

        category = np.argmax(kwargs['target'].cpu().data.numpy())

        # Calculate Gradient Replacement
        # if self.args.grad_replacement == 'class':
        #     pass
        # elif self.args.grad_replacement == 'instance':
        #     Zt = torch.mean(data).cpu().detach().numpy()
        #     Zt = torch.tensor(np.repeat(Zt, data.shape[0]), dtype=torch.float32)
        # elif self.args.grad_replacement == 'random':
        #     pass
        # elif self.args.grad_replacement == 'random_class':
        #     original_class_predictions = torch.max(kwargs['target']).item()

        if kwargs['save_perturbations']:
            self.perturbation_manager = PerturbationManager(
                original_signal=data.cpu().detach().numpy().flatten(),
                algo=self.args.algo, prediction_prob=np.max(kwargs['target'].cpu().data.numpy()),
                original_label=label, sample_id=self.args.single_sample_id)

        plt.plot(data, label="Original Signal Norm")
        gkernel = cv2.getGaussianKernel(3, 0.5)
        gaussian_blur_signal = cv2.filter2D(data.cpu().detach().numpy(), -1, gkernel).flatten()
        plt.plot(gaussian_blur_signal, label="Gaussian Blur")
        median_blur_signal = median_filter(data, 3)
        plt.plot(median_blur_signal, label="Median Blur")
        blurred_signal = (gaussian_blur_signal + median_blur_signal) / 2
        plt.plot(blurred_signal, label="Blurred Signal")
        # mask_init = np.ones(shape=len(data), dtype=np.float32)
        mask_init = np.full(len(data), 1e-1, dtype=np.float32)
        blurred_signal_norm = blurred_signal / np.max(blurred_signal)
        plt.plot(blurred_signal_norm, label="Blur Norm")

        # to log noise

        # if self.args.enable_noise:
        #     noise = np.zeros(original_signal.shape, dtype=np.float32)
        #     cv2.randn(noise, 0, 0.2)
        #     log_noise = original_signal
        #     log_noise+= noise
        #     log_noise = log_noise/np.max(log_noise)
        #     plt.plot(log_noise, label="Orig_Noise Norm")

        if self.enable_wandb:
            wandb.log({'Initialization': plt}, step=0)

        # # visualize the class imbalance plots if needed
        # g = sns.countplot(dataset.train_label)
        # g.set_xticklabels(['Train Class 0 ', 'Train Class 1'])
        # plt.show()

        # if ENABLE_WANDB:
        #     wandb.log({'Train Class Balanced vs Imbalanced': plt})

        # g = sns.countplot(dataset.test_label)
        # g.set_xticklabels(['Test Class 0 ', 'Test Class 1'])
        # plt.show()

        # if ENABLE_WANDB:
        #     wandb.log({'Test Class Balanced vs  Imbalanced': plt})

        blurred_signal = torch.tensor((blurred_signal_norm.reshape(1, -1)), dtype=torch.float32)
        mask = numpy_to_torch(mask_init, use_cuda=self.use_cuda)
        masks_init_ones = mask
        # todo: Ramesh - Upsample?
        # if use_cuda:
        #     upsample = torch.nn.UpsamplingBilinear2d(size=(224, 224)).cuda()
        # else:
        #     upsample = torch.nn.UpsamplingBilinear2d(size=(224, 224))

        optimizer = torch.optim.Adam([mask], lr=self.args.lr)

        if self.args.enable_lr_decay:
            scheduler = ExponentialLR(optimizer, gamma=self.args.lr_decay)

        print(f"{self.args.algo}: Category with highest probability {category}")
        print(f"{self.args.algo}: Optimizing.. ")

        metrics = {"TV Norm": [], 'TV Coeff': [], "MSE": [], "Budget": [], "Total Loss": [],
                   "Confidence": [],
                   "Saliency Var": [],
                   "Saliency Sum": [], "MSE Coeff": [], "L1 Coeff": [], "Mean Gradient": [],
                   "Category": [],
                   # "DTW": [], "DTW Coeff":[], 'EUC': [],
                   "epoch": {}, 'MSE Var': [], 'DIST': []}
        mse_loss_fn = torch.nn.MSELoss(reduction='mean')
        softdtw_loss_fn = SoftDTW()
        original_class_predictions = torch.max(kwargs['target']).item()

        # Static pick
        Zt = self.static_pick_zt(kwargs=kwargs, data=data, label=label)

        for i in range(self.args.max_itr):
            CUR_DIR = kwargs['save_dir'] + f'/epoch_{i}/'
            # os.system(f"mkdir -p {CUR_DIR}")

            # todo: Ramesh - Upsample?
            # upsampled_mask = upsample(mask)
            upsampled_mask = mask

            neg_upsampled = (1 - upsampled_mask)
            # cv2.imwrite(f"{CUR_DIR}/3-neg_upsampled.png", get_image(neg_upsampled.cpu().detach().numpy()))
            # ["zeros", "class_mean", "instance_mean", "random_instance", "random_opposing_instance"]

            if self.args.dynamic_replacement:
                Zt = self.dynamic_pick_zt(kwargs=kwargs, data=data, label=label)

            # perturbated_input = data.mul(upsampled_mask) + Zt.mul(1 - upsampled_mask)
            perturbated_input = data.mul(upsampled_mask) + Zt.mul(1 - upsampled_mask)

            if self.args.enable_blur:
                # perturbation = blurred_signal.mul(neg_upsampled)
                perturbation = blurred_signal.mul(upsampled_mask)
                perturbated_input += perturbation
                # cv2.imwrite(f"{CUR_DIR}/4-perturbation.png", get_image(perturbation.cpu().detach().numpy()))

            if self.args.enable_noise:
                noise = np.zeros(data.shape, dtype=np.float32)
                cv2.randn(noise, 0, 0.3)
                noise = numpy_to_torch(noise, use_cuda=self.use_cuda)
                perturbated_input += noise
                # cv2.imwrite(f"{CUR_DIR}/5-noise.png", get_image(noise.cpu().detach().numpy()))
            perturbated_input = perturbated_input.flatten()
            # cv2.imwrite(f"{CUR_DIR}/6-perturbed_input+noise.png",
            #             get_image(perturbated_input.cpu().detach().numpy()))
            # with torch.no_grad():
            outputs = self.softmax_fn(self.predict_fn(perturbated_input))

            metrics['Confidence'].append(float(outputs[category].item()))

            if kwargs['save_perturbations'] and not self.args.run_eval_every_epoch:
                self.perturbation_manager.add_perturbation(
                    perturbation=perturbated_input.cpu().detach().numpy().flatten(),
                    step=i, confidence=metrics['Confidence'][-1])
            c1 = self.args.mse_coeff * mse_loss_fn(outputs[1], kwargs['target'][1])
            c2 = self.args.l1_coeff * torch.mean(torch.abs(mask)) * float(self.args.enable_budget)
            c3 = self.args.tv_coeff * tv_norm(mask, self.args.tv_beta) * float(self.args.enable_tvnorm)
            mse_var = np.var(metrics['MSE']) if len(metrics['MSE']) > 1 else 0.0
            # -0.8452
            # c4 = self.args.dtw_coeff * softdtw_loss_fn(x=perturbated_input.reshape([1, -1]),
            #                                            y=Zt.reshape([1, -1])) * float(self.args.enable_dtw)
            dist_loss = torch.tensor(0.0)
            if self.args.enable_dist:
                if self.args.dist_loss == 'euc':
                    dist_loss = self.args.dist_coeff * mse_loss_fn(data.reshape([1, -1]),
                                                                   perturbated_input.reshape([1, -1]))
                elif self.args.dist_loss == 'w_euc':
                    dist_loss = self.args.dist_coeff * self.weighted_mse_loss(input=data.reshape([1, -1]),
                                                                              target=perturbated_input.reshape([1, -1]),
                                                                              weight=upsampled_mask.reshape([1, -1]))
                elif self.args.dist_loss == 'dtw':
                    dist_loss = self.args.dist_coeff * softdtw_loss_fn(x=perturbated_input.reshape([1, -1]),
                                                                       y=data.reshape([1, -1]))
                elif self.args.dist_loss == 'w_dtw':
                    dist_loss = self.args.dist_coeff * softdtw_loss_fn(x=perturbated_input.reshape([1, -1]),
                                                                       y=data.reshape([1, -1]),
                                                                       w=upsampled_mask.reshape([1, -1]))
                elif self.args.dist_loss == 'n_dtw':
                    dist_loss = self.args.dist_coeff * softdtw_loss_fn(x=perturbated_input.reshape([1, -1]),
                                                                       y=data.reshape([1, -1]),
                                                                       normalize=True)
                elif self.args.dist_loss == 'n_w_dtw':
                    dist_loss = self.args.dist_coeff * softdtw_loss_fn(x=perturbated_input.reshape([1, -1]),
                                                                       y=data.reshape([1, -1]),
                                                                       normalize=True,
                                                                       w=upsampled_mask.reshape([1, -1]))

            loss = c1 + c2 + c3 + dist_loss
            optimizer.zero_grad()
            loss.backward()
            metrics['Mean Gradient'].append(float(np.mean(mask.grad.cpu().detach().numpy())))
            metrics['TV Norm'].append(float(c3.item()))
            metrics['MSE'].append(float(c1.item()))
            metrics['DIST'].append(float(dist_loss.item()))
            metrics['Budget'].append(float(c2.item()))
            # metrics['DTW'].append(float(c4.item()))
            metrics['Total Loss'].append(float(loss.item()))
            metrics['Category'].append(int(np.argmax(outputs.cpu().detach().numpy())))
            metrics['Saliency Sum'].append(float(np.sum(mask.cpu().detach().numpy())))
            metrics['Saliency Var'].append(float(np.var(mask.cpu().detach().numpy())))
            metrics['MSE Coeff'].append(self.args.mse_coeff)
            metrics['TV Coeff'].append(self.args.tv_coeff)
            metrics['L1 Coeff'].append(self.args.l1_coeff)
            # metrics['DTW Coeff'].append(self.args.dtw_coeff)
            metrics['MSE Var'].append(mse_var)

            if self.args.early_stopping:
                if i > self.args.early_stop_criteria_patience and metrics['Confidence'][-1] <= (
                        self.args.early_stop_criteria_perc / 100) * original_class_predictions:
                    print(
                        f"Early Stop Criteria (Patience: {self.args.early_stop_criteria_patience}, Perc: {self.args.early_stop_criteria_perc}) met - Step: {i} | Confidence: {metrics['Confidence'][-1]}")
                    break

            print(
                f"Iter: {i}/{self.args.max_itr} | ({i / self.args.max_itr * 100:.2f}%) | LR: {optimizer.state_dict()['param_groups'][0]['lr']:.5f}"
                f"| MSE: {metrics['MSE'][-1]:.4f}, V:{mse_var:.4f}"
                f"| {self.args.dist_loss}: {metrics['DIST'][-1]:.4f}"
                f"| Budget: {metrics['Budget'][-1]:.4f} | Total Loss: {metrics['Total Loss'][-1]:.4f} "
                f"| Category: {metrics['Category'][-1]}"
                f"| Confidence: {metrics['Confidence'][-1]:.2f}"
                f"| S:{metrics['Saliency Sum'][-1]:.4f}"
                f"| RI: {self.r_index}",
                end="\r", flush=True
                # f"| DTW: {c4.item():.2f}"
            )

            optimizer.step()

            if self.args.enable_lr_decay:
                scheduler.step(epoch=i)

            # Optional: clamping seems to give better results
            if self.args.mask_norm == 'clamp':
                mask.data.clamp_(0, 1)
            elif self.args.mask_norm == 'sigmoid':
                mask.data = torch.nn.Sigmoid()(mask)
            elif self.args.mask_norm == 'softmax':
                mask.data = torch.nn.Softmax(dim=-1)(mask)
            elif self.args.mask_norm == 'none':
                pass
            mask.data.clamp_(0, 1)

            if self.args.run_eval_every_epoch:
                m = mask.cpu().detach().numpy().flatten()
                metrics["epoch"][f"epoch_{i}"] = {
                    'eval_metrics': run_evaluation_metrics(self.args.eval_replacement, kwargs['dataset'], data,
                                                           self.predict_fn, m,
                                                           kwargs['save_dir'], False)}

                if kwargs['save_perturbations']:
                    self.perturbation_manager.add_perturbation(
                        perturbation=perturbated_input.cpu().detach().numpy().flatten(),
                        saliency=mask.cpu().detach().numpy(),
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
                         **{"Gradient": wandb.Histogram(mask.grad.cpu().detach().numpy()),
                            "Training": [upsampled_mask, neg_upsampled, noise, perturbated_input,
                                         ] + [perturbation] if self.args.enable_blur else []
                            }
                         }
                if f"epoch_{i}" in metrics["epoch"]:
                    _mets = {**_mets, **metrics["epoch"][f"epoch_{i}"]['eval_metrics']}

                wandb.log(_mets)

            # cv2.imwrite(f"{CUR_DIR}/7-clamped_mask.png", get_image(mask.cpu().detach().numpy()))

        # todo Ramesh - Upsample?
        # upsampled_mask = upsample(mask)

        np.save(kwargs['save_dir'] + "/upsampled_mask", upsampled_mask.cpu().detach().numpy())
        np.save(kwargs['save_dir'] + "/mask", mask.cpu().detach().numpy())
        if self.enable_wandb:
            wandb.save(kwargs['save_dir'] + "/metrics.json")
            wandb.save(kwargs['save_dir'] + "/upsampled_mask.npy")
            wandb.save(kwargs['save_dir'] + "/mask.npy")

        mask = mask.squeeze(0).squeeze(0)
        mask = mask.cpu().detach().numpy().flatten()

        if self.enable_wandb:
            wandb.run.summary["pos_features"] = len(np.where(mask > 0)[0])
            wandb.run.summary["neg_features"] = len(np.where(mask < 0)[0])
        mask_min = np.min(mask)
        mask_max = np.max(mask)
        norm_mask = np.array([(x - mask_min) / (mask_max - mask_min) for x in mask])

        save_timeseries(mask=norm_mask, raw_mask=None, time_series=data.numpy(),
                        blurred=blurred_signal, save_dir=kwargs['save_dir'], enable_wandb=self.enable_wandb,
                        algo=self.args.algo, dataset=self.args.dataset, category=label )

        # Not a good idea
        # p = float(outputs[category].item())
        # mask_indices = {i: e for e, i in enumerate(np.argsort(mask)[::-1])}
        # sorted_mask = np.sort(mask)[::-1]
        # sorted_mask = sorted_mask[:math.ceil(len(mask) * p)]
        # sorted_mask = np.array([*sorted_mask, *np.zeros(len(mask) - len(sorted_mask))])
        # sorted_reverted_mask = list(range(len(mask)))
        # for i,e in mask_indices.items():
        #     sorted_reverted_mask[i]=sorted_mask[e]
        # mask = np.array(sorted_reverted_mask)
        # print(np.argsort(mask)[::-1])

        if self.enable_wandb:
            wandb.run.summary["norm_saliency_sum"] = np.sum(mask)
            wandb.run.summary["norm_saliency_var"] = np.var(mask)

        np.save("/tmp/r_mask", mask)
        np.save("/tmp/n_mask", norm_mask)
        return mask, self.perturbation_manager
