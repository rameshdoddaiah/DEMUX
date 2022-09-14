# -*- coding: utf-8 -*-
"""
| **@created on:** 9/26/20,
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
from scipy.ndimage.filters import gaussian_filter1d
from scipy.ndimage.filters import median_filter
import logging
import random
from nte.experiment.evaluation import run_evaluation_metrics
import seaborn as sns
from nte.utils.perturbation_manager import PerturbationManager

from nte.models.saliency_model import Saliency
from torch.optim.lr_scheduler import ExponentialLR, StepLR


class CMWRGradientSaliency(Saliency):
    def __init__(self, background_data, background_label, predict_fn, enable_wandb, use_cuda, args):
        super(CMWRGradientSaliency, self).__init__(background_data=background_data, background_label=background_label,
                                                 predict_fn=predict_fn)
        self.enable_wandb = enable_wandb
        self.use_cuda = use_cuda
        self.args = args
        self.softmax_fn = torch.nn.Softmax(dim=-1)

    def dynamic_pick_zt(self, kwargs, data, label):
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
                Zt = torch.tensor(self.background_data[random.randrange(0, len(self.background_data))],
                                  dtype=torch.float32)
            elif self.args.grad_replacement == 'random_opposing_instance':
                if label == 1:
                    sds = kwargs['dataset'].test_class_0_indices
                    sls = len(sds)
                else:
                    sds = kwargs['dataset'].test_class_1_indices
                    sls = len(sds)
                Zt = torch.tensor(sds[random.randrange(0, sls)], dtype=torch.float32)

            # a = original_signal.cpu().detach().numpy().flatten()
            # b = upsampled_mask.cpu().detach().numpy().flatten()
            # diff = np.array([np.linalg.norm(x - y) for x, y in zip(a, b)])
            # diff_norm = (diff - diff.min()) / (diff.max() - diff.min()) if np.sum(diff) > 0 else diff
            # p1 = original_signal.mul(upsampled_mask)
            # p2 = torch.tensor(diff_norm, dtype=torch.float32).mul(Mx)
            # perturbated_input = p1 + p2
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
                Zt = torch.tensor(self.background_data[random.randrange(0, len(self.background_data))],
                                  dtype=torch.float32)
            elif self.args.grad_replacement == 'random_opposing_instance':
                if label == 1:

                    Zt = torch.tensor(kwargs['dataset'].test_statistics['between_class']['opposing'][0],
                                      dtype=torch.float32)
                else:
                    Zt = torch.tensor(kwargs['dataset'].test_statistics['between_class']['opposing'][1],
                                      dtype=torch.float32)

            # a = original_signal.cpu().detach().numpy().flatten()
            # b = upsampled_mask.cpu().detach().numpy().flatten()
            # diff = np.array([np.linalg.norm(x - y) for x, y in zip(a, b)])
            # diff_norm = (diff - diff.min()) / (diff.max() - diff.min()) if np.sum(diff) > 0 else diff
            # p1 = original_signal.mul(upsampled_mask)
            # p2 = torch.tensor(diff_norm, dtype=torch.float32).mul(Mx)
            # perturbated_input = p1 + p2
        return Zt

    def generate_saliency(self, data, label, **kwargs):

        # with torch.no_grad():
        #     if self.args.bbm == 'rnn':
        #         target = self.softmax_fn(self.predict_fn(data.reshape(1, -1)))
        #     elif self.args.bbm == 'dnn':
        #         target = self.softmax_fn(self.predict_fn(data))
        #     else:
        #         raise Exception(f"Black Box model not supported: {self.args.bbm}")

        category = np.argmax(kwargs['target'].cpu().data.numpy())

        if kwargs['save_perturbations']:
            self.perturbation_manager = PerturbationManager(
                original_signal=data.cpu().detach().numpy().flatten(),
                algo=self.args.algo, prediction_prob=np.max(kwargs['target'].cpu().data.numpy()), original_label=label,
                sample_id=self.args.single_sample_id
            )

        plt.plot(data, label="Original Signal Norm")
        gkernel = cv2.getGaussianKernel(self.args.bwin, self.args.bsigma)
        gaussian_blur_signal = cv2.filter2D(data.cpu().detach().numpy(), -1, gkernel).flatten()
        plt.plot(gaussian_blur_signal, label="Gaussian Blur")
        median_blur_signal = median_filter(data, self.args.bwin)
        plt.plot(median_blur_signal, label="Median Blur")
        blurred_signal = (gaussian_blur_signal + median_blur_signal) / 2
        plt.plot(blurred_signal, label="Blurred Signal")
        divs = int(len(data) / (len(data) * self.args.sample * 0.01))
        # mask_init = np.ones(len(data), dtype=np.float32)
        mask_init = np.ones(divs, dtype=np.float32)
        blurred_signal_norm = blurred_signal / np.max(blurred_signal)
        plt.plot(blurred_signal_norm, label="Blur Norm")

        if self.enable_wandb:
            wandb.log({'Initialization': plt}, step=0)

        blurred_signal = torch.tensor((blurred_signal_norm.reshape(1, -1)), dtype=torch.float32)
        mask = numpy_to_torch(mask_init, use_cuda=self.use_cuda)
        masks_init_ones = mask

        optimizer = torch.optim.Adam([mask], lr=self.args.lr)

        if self.args.enable_lr_decay:
            scheduler = ExponentialLR(optimizer, gamma=self.args.lr_decay)

        print(f"{self.args.algo}: Category with highest probability {category}")
        print(f"{self.args.algo}: Optimizing.. ")

        metrics = {"TV Norm": [], "Budget": [], "Total Loss": [], "Confidence": [], "Saliency Var": [],
                   "Saliency Sum": [], "TV Coeff": [], "L1 Coeff": [], "Mean Gradient": [],
                   "Category": [], "epoch": {}}

        # upsample = torch.nn.UpsamplingBilinear2d(size=(224, 224))
        # upsample = torch.nn.Upsample(scale_factor=2, mode='nearest')
        # upsample = torch.nn.Upsample(scale_factor=2, mode='bilinear')
        upsample = torch.nn.Upsample(size=len(data), mode='linear')

        # Static pick
        Zt = self.static_pick_zt(kwargs=kwargs, data=data, label=label)

        for i in tqdm.tqdm(range(self.args.max_itr)):
            CUR_DIR = kwargs['save_dir'] + f'/epoch_{i}/'
            os.system(f"mkdir -p {CUR_DIR}")

            # todo: Ramesh - Upsample?
            upsampled_mask = upsample(mask)
            # upsampled_mask = mask

            neg_upsampled = (1 - upsampled_mask)
            # cv2.imwrite(f"{CUR_DIR}/3-neg_upsampled.png", get_image(neg_upsampled.cpu().detach().numpy()))
            Zt = self.dynamic_pick_zt(kwargs=kwargs, data=data, label=label)

            perturbated_input = data.mul(upsampled_mask) + Zt.mul(1-upsampled_mask)
            # perturbated_input = data.mul(upsampled_mask) + Zt.mul(1 - upsampled_mask)

            if self.args.enable_blur:
                perturbation = blurred_signal.mul(neg_upsampled)
            else:
                perturbation = 0

            perturbated_input += perturbation
            # cv2.imwrite(f"{CUR_DIR}/4-perturbation.png", get_image(perturbation.cpu().detach().numpy()))

            # if self.args.enable_noise:
            noise = np.zeros(data.shape, dtype=np.float32)
            cv2.randn(noise, 0, 0.3)
            noise = numpy_to_torch(noise, use_cuda=self.use_cuda)
            perturbated_input += noise
            # cv2.imwrite(f"{CUR_DIR}/5-noise.png", get_image(noise.cpu().detach().numpy()))

            # cv2.imwrite(f"{CUR_DIR}/6-perturbed_input+noise.png",
            #             get_image(perturbated_input.cpu().detach().numpy()))
            # with torch.no_grad():
            outputs = self.softmax_fn(self.predict_fn(perturbated_input))
            c1 = self.args.l1_coeff * torch.mean(torch.abs(1 - mask)) * float(self.args.enable_budget)
            c2 = self.args.tv_coeff * tv_norm(mask, self.args.tv_beta) * float(self.args.enable_tvnorm)
            c3 = outputs[0, 0, category]
            loss = c1 + c2 + c3
            optimizer.zero_grad()
            loss.backward()

            metrics['Mean Gradient'].append(float(np.mean(mask.grad.cpu().detach().numpy())))
            metrics['TV Norm'].append(float(c2.item()))
            metrics['Budget'].append(float(c1.item()))
            metrics['Confidence'].append(float(c3.item()))
            metrics['Total Loss'].append(float(loss.item()))
            metrics['Category'].append(int(np.argmax(outputs.cpu().detach().numpy())))
            metrics['Saliency Sum'].append(float(np.sum(1 - mask.cpu().detach().numpy())))
            metrics['Saliency Var'].append(float(np.var(1 - mask.cpu().detach().numpy())))
            metrics['TV Coeff'].append(self.args.tv_coeff)
            metrics['L1 Coeff'].append(self.args.l1_coeff)

            print(
                f"Iter: {i}/{self.args.max_itr} | ({i / self.args.max_itr * 100:.2f}%) | LR: {optimizer.state_dict()['param_groups'][0]['lr']:.5f}"
                f"| TV_Norm: {metrics['TV Norm'][-1]:.4f} "
                f"| Budget: {metrics['Budget'][-1]:.4f} | Total Loss: {metrics['Total Loss'][-1]:.4f} "
                f"| Category: {metrics['Category'][-1]}"
                f"| Confidence: {metrics['Confidence'][-1]:.2f}")
            optimizer.step()

            if self.args.enable_lr_decay:
                scheduler.step(epoch=i)

            # Optional: clamping seems to give better results
            mask.data.clamp_(0, 1)

            if self.args.run_eval_every_epoch:
                m = (1 - upsample(mask)).cpu().detach().numpy().flatten()
                metrics["epoch"][f"epoch_{i}"] = {
                    'eval_metrics': run_evaluation_metrics(self.args.eval_replacement, kwargs['dataset'], data,
                                                           self.predict_fn, m,
                                                           kwargs['save_dir'], False)}

                if kwargs['save_perturbations']:
                    self.perturbation_manager.add_perturbation(
                        perturbation=perturbated_input.cpu().detach().numpy().flatten(),
                        step=i, confidence=metrics['Confidence'][-1],
                        saliency=m,
                        insertion=metrics["epoch"][f"epoch_{i}"]['eval_metrics']['Insertion']['trap'],
                        deletion=metrics["epoch"][f"epoch_{i}"]['eval_metrics']['Deletion']['trap'],
                        final_auc=metrics["epoch"][f"epoch_{i}"]['eval_metrics']['Final']['AUC'],
                        mean_gradient=metrics['Mean Gradient'][-1],
                        tv_norm=metrics['TV Norm'][-1],
                        main_loss=metrics["Confidence"][-1],
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

        upsampled_mask = 1 - upsampled_mask

        np.save(kwargs['save_dir'] + "/upsampled_mask", upsampled_mask.cpu().detach().numpy())
        np.save(kwargs['save_dir'] + "/mask", mask.cpu().detach().numpy())
        if self.enable_wandb:
            wandb.save(kwargs['save_dir'] + "/metrics.json")
            wandb.save(kwargs['save_dir'] + "/upsampled_mask.npy")
            wandb.save(kwargs['save_dir'] + "/mask.npy")
        # save((1 - upsampled_mask), original_signal, blurred_signal, SAVE_DIR, ENABLE_WANDB)
        save_timeseries(mask=mask, time_series=data,
                        blurred=blurred_signal, save_dir=kwargs['save_dir'], enable_wandb=self.enable_wandb,
                        algo=self.args.algo, dataset=self.args.dataset)

        mask = upsample(mask)
        mask = mask.squeeze(0).squeeze(0)
        mask = mask.cpu().detach().numpy().flatten()
        return 1 - mask, self.perturbation_manager
