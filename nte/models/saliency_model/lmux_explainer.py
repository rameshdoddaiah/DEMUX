# -*- coding: utf-8 -*-
"""
| **@created on:** 9/28/20,
| **@author:** prathyushsp,
| **@version:** v0.0.1
|
| **Description:**
| **Sphinx Documentation Status:**
"""
import torch
import cv2
import numpy as np
import wandb
from nte.experiment.utils import tv_norm
from nte.utils.perturbation_manager import PerturbationManager
from nte.models.saliency_model import Saliency
from torch.optim.lr_scheduler import ExponentialLR
from nte.utils.priority_buffer import PrioritizedBuffer
import scipy
import torch
from torch.autograd import Variable
from collections import defaultdict
from rainbow_print import printr


class LMUXExplainer(Saliency):
    def __init__(self, background_data, background_label, predict_fn, enable_wandb, use_cuda, args):
        super(LMUXExplainer, self).__init__(background_data=background_data,
                                            background_label=background_label,
                                            predict_fn=predict_fn)
        self.enable_wandb = enable_wandb
        self.use_cuda = use_cuda
        self.args = args
        self.softmax_fn = torch.nn.Softmax(dim=-1)
        self.perturbation_manager = None
        self.rs_priority_buffer = None
        self.ro_priority_buffer = None
        self.eps = None
        self.eps_decay = 0.9991  # 0.9996#0.997

    def priority_dual_greedy_pick_rt(self, kwargs, data, label):
        self.eps *= self.eps_decay
        if np.random.uniform() < self.eps:
            self.mode = 'Explore'
            rs_index = [np.random.choice(len(getattr(self.args.dataset, f"test_class_{int(label)}_data")))]
            Rs, rs_weight = [getattr(self.args.dataset, f"test_class_{int(label)}_data")[rs_index[0]]][0], [1.0]
            ro_index = [np.random.choice(list(range(0, len(self.ro_priority_buffer.memory))))]
            Ro, ro_weight = self.ro_priority_buffer.memory[ro_index[0]], [1.0]
        else:
            self.mode = 'Exploit'
            [Rs], rs_weight, rs_index = self.rs_priority_buffer.sample(1)
            [Ro], ro_weight, ro_index = self.ro_priority_buffer.sample(1)
        return {'rs': [Rs, rs_weight, rs_index],
                'ro': [Ro, ro_weight, ro_index]}

    def euc(self, point, dist, cov=None):
        return scipy.spatial.distance.euclidean(point, np.mean(dist, axis=0))

    def generate_saliency(self, data, label, **kwargs):

        if isinstance(data, np.ndarray):
            data = torch.tensor(data, dtype=torch.float32)

        target_prediction_probabilities = kwargs['target'].cpu().data.numpy()
        top_prediction_class = np.argmax(kwargs['target'].cpu().data.numpy())
        top_prediction_confidence = target_prediction_probabilities[top_prediction_class]
        num_classes = getattr(kwargs['dataset'], f"num_classes")

        # Initialize rs_priority_buffer
        self.rs_priority_buffer = PrioritizedBuffer(background_data=getattr(
            kwargs['dataset'], f"test_class_{int(top_prediction_class)}_data"))

        # Initialize ro_priority_buffer
        if num_classes > 2:
            class_choices = list(range(0, int(top_prediction_class))) + \
                list(range(int(top_prediction_class)+1, num_classes))
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
                centroids = getattr(kwargs['dataset'], "cluster_meta")["centroids"]
                dist = [scipy.spatial.distance.euclidean(
                    data.numpy().reshape(1, -1), centroids[c]) for c in range(getattr(kwargs['dataset'], f"cluster_meta")["n_clusters"])]
                indices = getattr(kwargs['dataset'], "cluster_meta")[f"cluster_{np.argmax(dist)}_indices"]
                self.ro_priority_buffer = PrioritizedBuffer(background_data=np.take(
                    getattr(kwargs['dataset'], "train_data"), indices[0], axis=0))
        else:
            self.ro_priority_buffer = PrioritizedBuffer(background_data=getattr(
                kwargs['dataset'], f"test_class_{1 - int(label)}_data"))

        self.eps = 1.0

        # Declare loss functions
        l_preserve = torch.nn.KLDivLoss(size_average=False)
        l_ssd = torch.nn.MSELoss(reduction='mean')  # nn.CosineSimilarity(dim=1, eps=1e-6)
        l_budget = None
        l_tv_norm = None

        # Initialize Mask of shape [Classes, Timesteps]
        mask_init = np.random.uniform(size=[num_classes, data.shape[-1]], low=-1e-2, high=1e-2)
        n_mask = Variable(torch.from_numpy(mask_init), requires_grad=True)
        assert n_mask.shape.numel() == num_classes * data.shape[-1]

        # Setup optimizer
        optimizer = torch.optim.Adam([n_mask], lr=self.args.lr)

        if self.args.enable_lr_decay:
            scheduler = ExponentialLR(optimizer, gamma=self.args.lr_decay)

        if kwargs['save_perturbations']:
            self.perturbation_manager = PerturbationManager(original_signal=data.cpu().detach().numpy().flatten(), algo=self.args.algo, prediction_prob=np.max(kwargs['target'].cpu().data.numpy()),
                                                            original_label=label, sample_id=self.args.single_sample_id)

        print(f"{self.args.algo}: Optimizing... ")
        metrics = defaultdict(lambda: [])

        # Training
        i, mask_repo = 0, {}
        while i <= self.args.max_itr:
            # Sample Ro and Rs
            picks = self.priority_dual_greedy_pick_rt(kwargs=kwargs, data=data, label=top_prediction_class)
            rs, rs_weight, rs_index = picks['rs']
            ro, ro_weight, ro_index = picks['ro']
            Rt = np.zeros(shape=n_mask.shape)

            assert rs.shape == ro.shape
            assert rs.shape == n_mask[0].shape

            # Prepare Rt
            for c in range(num_classes):
                for e, (rs_e, ro_e, m) in enumerate(zip(rs.flatten(), ro.flatten(), n_mask[c].detach().numpy().flatten())):
                    Rt[c][e] = rs_e if m < 0 else ro_e
            Rt = torch.tensor(Rt, dtype=torch.float32)

            assert Rt.shape == n_mask.shape

            # Create Perturbation
            perturbated_input = data.mul(n_mask) + Rt.mul(1-n_mask)

            assert perturbated_input.shape == n_mask.shape

            # Add Noise
            if self.args.enable_noise:
                noise = np.zeros(shape=n_mask.shape, dtype=np.float32)
                cv2.randn(noise, 0, 0.3)
                perturbated_input += torch.from_numpy(noise)

            # Fetch the predictions
            # with torch.no_grad():
            outputs = self.softmax_fn(self.predict_fn(perturbated_input.float()))
            # print(n_mask.requires_grad)
            if kwargs['save_perturbations'] and not self.args.run_eval_every_epoch:
                self.perturbation_manager.add_perturbation(perturbation=perturbated_input.cpu(
                ).detach().numpy().flatten(), step=i, confidence=metrics['Confidence'][-1])

            # L Preservation Loss
            l_prev_loss = 0.0
            if self.args.l_prev == "kld":
                for c in range(num_classes):
                    l_prev_loss += l_preserve(outputs[c].log(), kwargs['target'])
            elif self.args.l_prev == "mse":
                for c in range(num_classes):
                    l_prev_loss += l_preserve(outputs[c], kwargs['target'])

            # L Budget Loss
            l_budget_loss = 0.0
            for c in range(num_classes):
                l_budget_loss += torch.mean(torch.abs(n_mask[c])) * float(self.args.enable_budget)

            # L TV Norm Loss
            l_tv_norm_loss = 0.0
            for c in range(num_classes):
                l_tv_norm_loss += tv_norm(n_mask[c], self.args.tv_beta) * float(self.args.enable_tvnorm)

            # L SSD Loss
            top_class_mask = n_mask[top_prediction_class].clone()
            distinct_mask = torch.zeros_like(n_mask[top_prediction_class])
            for c in range(num_classes):
                if c != top_prediction_class:
                    distinct_mask += (top_class_mask - (n_mask[c]*target_prediction_probabilities[c]))
                    # distinct_mask += (top_class_mask - (n_mask[c]*1.0))
            dmin, dmax = np.min(np.array(distinct_mask.data)), np.max(np.array(distinct_mask.data))
            distinct_mask = 2 * ((distinct_mask-dmin)/(dmax-dmin + 1e-27)) - 1
            l_ssd_loss = l_ssd(n_mask[top_prediction_class], distinct_mask)

            # Calculate Total Loss
            loss = (self.args.l_prev_coeff * l_prev_loss)+(self.args.l_budget_coeff * l_budget_loss) + \
                (self.args.l_tv_norm_coeff * l_tv_norm_loss)+(self.args.l_ssd_coeff * l_ssd_loss)

            # Calculate Priority Loss
            rs_prios = loss * (rs_weight[0])*0.0 if loss < 0 else loss * (rs_weight[0])
            ro_prios = loss * (ro_weight[0])*0.0 if loss < 0 else loss * (ro_weight[0])

            # Update Priority Loss
            loss = loss * (rs_weight[0] + ro_weight[0]) / 2

            # if i >= self.args.max_itr:
            #     mask_repo[float(l_prev_loss.item())] = n_mask[top_prediction_class].cpu().detach().numpy().flatten()
                # mask_repo[float(outputs[top_prediction_class][top_prediction_class].item())
                #           ] = n_mask[top_prediction_class].cpu().detach().numpy().flatten()
                # if l_prev_loss <= 0.01:
                #     break

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()

            metrics['Mean Gradient'].append(float(np.mean(n_mask.grad[top_prediction_class].cpu().detach().numpy())))
            metrics['L_Preserve'].append(float(l_prev_loss.item()))
            metrics['L_Budget'].append(float(l_budget_loss.item()))
            metrics['L_TV_Norm'].append(float(l_tv_norm_loss.item()))
            metrics['L_SSD'].append(float(l_ssd_loss.item()))
            metrics['L_Total'].append(float(loss.item()))
            metrics['Category'].append(int(np.argmax(outputs[top_prediction_class].cpu().detach().numpy())))
            metrics['Saliency Sum'].append(float(np.mean(np.abs(n_mask[top_prediction_class].cpu().detach().numpy()))))
            metrics['Saliency Var'].append(float(np.var(n_mask[top_prediction_class].cpu().detach().numpy())))
            metrics['L_Preserve Var'].append(np.var(metrics['L_Preserve']) if len(metrics['L_Preserve']) > 1 else 0.0)
            metrics['L_Budget Var'].append(np.var(metrics['L_Budget']) if len(metrics['L_Budget']) > 1 else 0.0)
            metrics['L_TV_Norm Var'].append(np.var(metrics['L_TV_Norm']) if len(metrics['L_TV_Norm']) > 1 else 0.0)
            metrics['L_SSD Var'].append(np.var(metrics['L_SSD']) if len(metrics['L_SSD']) > 1 else 0.0)
            metrics['L_Total Var'].append(np.var(metrics['L_Total']) if len(metrics['L_Total']) > 1 else 0.0)
            metrics['Confidence'].append(float(outputs[top_prediction_class][top_prediction_class].item()))

            if self.args.print == 'yes':
                m_data = (f'{i}/{self.args.max_itr}({i / self.args.max_itr * 100:.2f}%)'
                        f' | P:{metrics["L_Preserve"][-1]: .4f}(V:{metrics["L_Preserve Var"][-1]:.4f})'
                        # f' | B:{metrics["L_Budget"][-1]: .4f}(V:{metrics["L_Budget Var"][-1]: .4f})'
                        # f' | TV:{metrics["L_TV_Norm"][-1]: .4f}(V: {metrics["L_TV_Norm Var"][-1]: .4f})'
                        f' | S:{metrics["L_SSD"][-1]: .4f}(V:{metrics["L_SSD Var"][-1]: .4f})'
                        f' | TL:{metrics["L_Total"][-1]: .4f}(V:{metrics["L_Total Var"][-1]: .4f})'
                        f' | EPS:{self.eps:.4f} , M:{self.mode} | TCP:{outputs[top_prediction_class][top_prediction_class]: .4f} ({outputs[top_prediction_class][top_prediction_class]/top_prediction_confidence*100:.2f}%)')
                printr(m_data, sep='|')

            optimizer.step()

            self.rs_priority_buffer.update_priorities(rs_index, [rs_prios.item()])
            self.ro_priority_buffer.update_priorities(ro_index, [ro_prios.item()])

            if self.args.enable_lr_decay:
                scheduler.step(epoch=i)

            # Clamp mask
            n_mask.data.clamp_(-1, 1)

            if self.enable_wandb:
                _mets = {**{k: v[-1] for k, v in metrics.items() if k != "epoch"}}
                if f"epoch_{i}" in metrics["epoch"]:
                    _mets = {**_mets, **metrics["epoch"][f"epoch_{i}"]['eval_metrics']}
                wandb.log(_mets)

            i += 1

        # Fetch Top Class Mask
        mask = n_mask[top_prediction_class].cpu().detach().numpy().flatten()
        # mask = np.mean(np.array([mask_repo[m] for m in sorted(mask_repo)[0:10]]), axis=0)

        # Normalize Mask
        norm_mask = (mask-mask.min())/(mask.max()-mask.min())

        # return norm_mask, self.perturbation_manager, n_mask.cpu().detach().numpy().flatten()
        return mask, self.perturbation_manager, n_mask.cpu().detach().numpy()


# """
# import torch
# import cv2
# import numpy as np
# import wandb
# from nte.experiment.utils import tv_norm
# from nte.utils.perturbation_manager import PerturbationManager
# from nte.models.saliency_model import Saliency
# from torch.optim.lr_scheduler import ExponentialLR
# from nte.utils.priority_buffer import PrioritizedBuffer
# import scipy
# import torch
# from torch.autograd import Variable
# from collections import defaultdict
# from rainbow_print import printr
# import copy


# class LMUXExplainer(Saliency):
#     def __init__(self, background_data, background_label, predict_fn, enable_wandb, use_cuda, args):
#         super(LMUXExplainer, self).__init__(background_data=background_data,
#                                             background_label=background_label,
#                                             predict_fn=predict_fn)
#         self.enable_wandb = enable_wandb
#         self.use_cuda = use_cuda
#         self.args = args
#         self.softmax_fn = torch.nn.Softmax(dim=-1)
#         self.perturbation_manager = None
#         self.rs_priority_buffer = None
#         self.ro_priority_buffer = None
#         self.eps = None
#         self.eps_decay = 0.9991  # 0.9996#0.997

#     def priority_dual_greedy_pick_rt(self, kwargs, data, label):
#         self.eps *= self.eps_decay
#         if np.random.uniform() < self.eps:
#             self.mode = 'Explore'
#             rs_index = [np.random.choice(len(getattr(self.args.dataset, f"test_class_{int(label)}_data")))]
#             Rs, rs_weight = [getattr(self.args.dataset, f"test_class_{int(label)}_data")[rs_index[0]]][0], [1.0]
#             ro_index = [np.random.choice(list(range(0, len(self.ro_priority_buffer.memory))))]
#             Ro, ro_weight = self.ro_priority_buffer.memory[ro_index[0]], [1.0]
#         else:
#             self.mode = 'Exploit'
#             [Rs], rs_weight, rs_index = self.rs_priority_buffer.sample(1)
#             [Ro], ro_weight, ro_index = self.ro_priority_buffer.sample(1)
#         return {'rs': [Rs, rs_weight, rs_index],
#                 'ro': [Ro, ro_weight, ro_index]}

#     def euc(self, point, dist, cov=None):
#         return scipy.spatial.distance.euclidean(point, np.mean(dist, axis=0))

#     def generate_saliency(self, data, label, **kwargs):

#         if isinstance(data, np.ndarray):
#             data = torch.tensor(data, dtype=torch.float32)

#         target_prediction_probabilities = kwargs['target'].cpu().data.numpy()
#         top_prediction_class = np.argmax(kwargs['target'].cpu().data.numpy())
#         # top_prediction_class2 = np.argmax(kwargs['target'].cpu().data.numpy())
#         top_prediction_confidence = target_prediction_probabilities[top_prediction_class]
#         num_classes = getattr(kwargs['dataset'], f"num_classes")

#         n_mask2 = []
#         # for i in range(num_classes):
#         for i in range(1):
#             # top_prediction_class = i
#             # Initialize rs_priority_buffer
#             self.rs_priority_buffer = PrioritizedBuffer(background_data=getattr(
#                 kwargs['dataset'], f"test_class_{int(top_prediction_class)}_data"))

#             # Initialize ro_priority_buffer
#             if num_classes > 2:
#                 class_choices = list(range(0, int(top_prediction_class))) + \
#                     list(range(int(top_prediction_class)+1, num_classes))
#                 if self.args.p_buf_init == "random":
#                     rand_label = np.random.choice(class_choices)
#                     self.ro_priority_buffer = PrioritizedBuffer(background_data=getattr(
#                         kwargs['dataset'], f"test_class_{rand_label}_data"))
#                 elif self.args.p_buf_init == "maha":
#                     dist = [self.mahalanobis(data.numpy().reshape(1, -1), getattr(
#                         kwargs['dataset'], f"test_class_{c}_data")) for c in class_choices]
#                     self.ro_priority_buffer = PrioritizedBuffer(background_data=getattr(
#                         kwargs['dataset'], f"test_class_{np.argmax(dist)}_data"))
#                 elif self.args.p_buf_init == "euc":
#                     dist = [self.euc(data.numpy().reshape(1, -1), getattr(
#                         kwargs['dataset'], f"test_class_{c}_data")) for c in class_choices]
#                     self.ro_priority_buffer = PrioritizedBuffer(background_data=getattr(
#                         kwargs['dataset'], f"test_class_{np.argmax(dist)}_data"))
#                 elif self.args.p_buf_init == "cluster":
#                     centroids = getattr(kwargs['dataset'], "cluster_meta")["centroids"]
#                     dist = [scipy.spatial.distance.euclidean(
#                         data.numpy().reshape(1, -1), centroids[c]) for c in range(getattr(kwargs['dataset'], f"cluster_meta")["n_clusters"])]
#                     indices = getattr(kwargs['dataset'], "cluster_meta")[f"cluster_{np.argmax(dist)}_indices"]
#                     self.ro_priority_buffer = PrioritizedBuffer(background_data=np.take(
#                         getattr(kwargs['dataset'], "train_data"), indices[0], axis=0))
#             else:
#                 self.ro_priority_buffer = PrioritizedBuffer(background_data=getattr(
#                     kwargs['dataset'], f"test_class_{1 - int(label)}_data"))

#             self.eps = 1.0

#             # Declare loss functions
#             l_preserve = torch.nn.KLDivLoss(size_average=False)
#             # l_ssd = torch.nn.KLDivLoss(size_average=False)
#             l_ssd = torch.nn.MSELoss(reduction='mean') 
#             # l_ssd = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
#             l_budget = None
#             l_tv_norm = None

#             # Initialize Mask of shape [Classes, Timesteps]
#             mask_init = np.random.uniform(size=[num_classes, data.shape[-1]], low=-1e-2, high=1e-2)
#             n_mask = Variable(torch.from_numpy(mask_init), requires_grad=True)
#             assert n_mask.shape.numel() == num_classes * data.shape[-1]

#             # Setup optimizer
#             optimizer = torch.optim.Adam([n_mask], lr=self.args.lr)

#             if self.args.enable_lr_decay:
#                 scheduler = ExponentialLR(optimizer, gamma=self.args.lr_decay)

#             if kwargs['save_perturbations']:
#                 self.perturbation_manager = PerturbationManager(original_signal=data.cpu().detach().numpy().flatten(), algo=self.args.algo, prediction_prob=np.max(kwargs['target'].cpu().data.numpy()),
#                                                                 original_label=label, sample_id=self.args.single_sample_id)

#             print(f"{self.args.algo}: Optimizing... ")
#             metrics = defaultdict(lambda: [])

#             # Training
#             i, mask_repo = 0, {}
#             while i <= self.args.max_itr:
#                 # Sample Ro and Rs
#                 picks = self.priority_dual_greedy_pick_rt(kwargs=kwargs, data=data, label=top_prediction_class)
#                 rs, rs_weight, rs_index = picks['rs']
#                 ro, ro_weight, ro_index = picks['ro']
#                 Rt = np.zeros(shape=n_mask.shape)

#                 assert rs.shape == ro.shape
#                 assert rs.shape == n_mask[0].shape

#                 # Prepare Rt
#                 for c in range(num_classes):
#                     for e, (rs_e, ro_e, m) in enumerate(zip(rs.flatten(), ro.flatten(), n_mask[c].detach().numpy().flatten())):
#                         Rt[c][e] = rs_e if m < 0 else ro_e
#                 Rt = torch.tensor(Rt, dtype=torch.float32)

#                 assert Rt.shape == n_mask.shape

#                 # Create Perturbation
#                 perturbated_input = data.mul(n_mask) + Rt.mul(1-n_mask)

#                 assert perturbated_input.shape == n_mask.shape

#                 # Add Noise
#                 if self.args.enable_noise:
#                     noise = np.zeros(shape=n_mask.shape, dtype=np.float32)
#                     cv2.randn(noise, 0, 0.3)
#                     perturbated_input += torch.from_numpy(noise)

#                 # Fetch the predictions
#                 # with torch.no_grad():
#                 outputs = self.softmax_fn(self.predict_fn(perturbated_input.float()))
#                 # print(n_mask.requires_grad)
#                 if kwargs['save_perturbations'] and not self.args.run_eval_every_epoch:
#                     self.perturbation_manager.add_perturbation(perturbation=perturbated_input.cpu().detach().numpy().flatten(), step=i, confidence=metrics['Confidence'][-1])

#                 # L Preservation Loss
#                 l_prev_loss = 0.0
#                 if self.args.l_prev == "kld":
#                     for c in range(num_classes):
#                         l_prev_loss += l_preserve(outputs[c].log(), kwargs['target'])
#                         # if c == top_prediction_class:
#                         #     if c == top_prediction_class2:
#                         #         l_prev_loss += l_preserve(outputs[c].log(), kwargs['target'])
#                         #     else:
#                         #         l_prev_loss += outputs[c][c]

#                 elif self.args.l_prev == "mse":
#                     for c in range(num_classes):
#                         l_prev_loss += l_preserve(outputs[c], kwargs['target'])
#                         # if c == top_prediction_class:
#                         #     if c == top_prediction_class2:
#                         #         l_prev_loss += l_preserve(outputs[c], kwargs['target'])
#                         #     else:
#                         #         l_prev_loss += outputs[c]

#                 # L Budget Loss
#                 l_budget_loss = 0.0
#                 for c in range(num_classes):
#                     l_budget_loss += torch.mean(torch.abs(n_mask[c])) * float(self.args.enable_budget)
#                     # if c == top_prediction_class:
#                     #     l_budget_loss += torch.mean(torch.abs(n_mask[c])) * float(self.args.enable_budget)

#                 # L TV Norm Loss
#                 l_tv_norm_loss = 0.0
#                 for c in range(num_classes):
#                     l_tv_norm_loss += tv_norm(n_mask[c], self.args.tv_beta) * float(self.args.enable_tvnorm)
#                     # if c == top_prediction_class:
#                     #     l_tv_norm_loss += tv_norm(n_mask[c], self.args.tv_beta) * float(self.args.enable_tvnorm)

#                 # L SSD Loss

#                 l_ssd_loss = torch.tensor(0.0)
#                 # if top_prediction_class == top_prediction_class2:
#                 # top_class_mask = n_mask[top_prediction_class]
#                 top_class_mask = n_mask[top_prediction_class].clone()
#                 # distinct_mask = torch.zeros_like(n_mask[top_prediction_class])
#                 distinct_mask = n_mask[top_prediction_class].clone()
#                 #distinct top_class -  each class first, don't add or increase top_class saliency
#                 #distinct  only do distinact_mask computation for the intersected boolean time steps and not for all time steps in the series
#                 #ask PP why loss doesn't go down 
#                 # temp_saliency = copy.deepcopy(n_mask[top_prediction_class].cpu().detach().numpy())
#                 # temp_n_masks = copy.deepcopy(n_mask.cpu().detach().numpy())
#                 # temp_saliency[temp_saliency >= alpha] = 1
#                 # temp_saliency[temp_saliency < alpha] = 0
#                 # temp_saliency = temp_saliency.astype(int)

#                 # for k in range(len(temp_n_masks)):
#                 #     temp_n_masks[k][temp_n_masks[k] >= alpha] = 1
#                 #     temp_n_masks[k][temp_n_masks[k] < alpha] = 0

#                 # temp_n_masks = np.array(temp_n_masks)
#                 # temp_n_masks = temp_n_masks.astype(int)

#                 # for i in range(num_classes):
#                 #     if i != top_prediction_class:
#                 alpha = 0.5
#                 for c in range(num_classes):
#                     if c != top_prediction_class:
#                         distinct_mask = distinct_mask - n_mask[c]*target_prediction_probabilities[c]
#                         # intersection = temp_saliency & temp_n_masks[c] 
#                         # distinct_mask += top_class_mask - n_mask[c]*target_prediction_probabilities[c]
#                         # for d in range(data.shape[1]):
#                         #     # if intersection[d]:
#                         #     if top_class_mask[d] > alpha and n_mask[c][d] > alpha:
#                         #         distinct_mask[d] = distinct_mask[d] + top_class_mask[d] - n_mask[c][d]
#                         #     else:
#                         #         distinct_mask[d] = distinct_mask[d] + top_class_mask[d] 

#                 dmin, dmax = np.min(np.array(distinct_mask.data)), np.max(np.array(distinct_mask.data))
#                 den = dmax-dmin
#                 if den == 0:
#                     den = 1e-27
#                 distinct_mask = 2 * ((distinct_mask-dmin)/(den)) - 1
#                 l_ssd_loss = l_ssd(n_mask[top_prediction_class], distinct_mask)

#                 # Calculate Total Loss
#                 loss = (self.args.l_prev_coeff * l_prev_loss)+(self.args.l_budget_coeff * l_budget_loss) + \
#                     (self.args.l_tv_norm_coeff * l_tv_norm_loss)+(self.args.l_ssd_coeff * l_ssd_loss)

#                 # Calculate Priority Loss
#                 rs_prios = loss * (rs_weight[0])*0.0 if loss < 0 else loss * (rs_weight[0])
#                 ro_prios = loss * (ro_weight[0])*0.0 if loss < 0 else loss * (ro_weight[0])

#                 # Update Priority Loss
#                 loss = loss * (rs_weight[0] + ro_weight[0]) / 2

#                 # if i >= self.args.max_itr:
#                 #     mask_repo[float(l_prev_loss.item())] = n_mask[top_prediction_class].cpu().detach().numpy().flatten()
#                     # mask_repo[float(outputs[top_prediction_class][top_prediction_class].item())
#                     #           ] = n_mask[top_prediction_class].cpu().detach().numpy().flatten()
#                     # if l_prev_loss <= 0.01:
#                     #     break

#                 # Backpropagation
#                 optimizer.zero_grad()
#                 loss.backward()

#                 metrics['Mean Gradient'].append(float(np.mean(n_mask.grad[top_prediction_class].cpu().detach().numpy())))
#                 metrics['L_Preserve'].append(float(l_prev_loss.item()))
#                 metrics['L_Budget'].append(float(l_budget_loss.item()))
#                 metrics['L_TV_Norm'].append(float(l_tv_norm_loss.item()))
#                 metrics['L_SSD'].append(float(l_ssd_loss.item()))
#                 metrics['L_Total'].append(float(loss.item()))
#                 metrics['Category'].append(int(np.argmax(outputs[top_prediction_class].cpu().detach().numpy())))
#                 metrics['Saliency Sum'].append(float(np.mean(np.abs(n_mask[top_prediction_class].cpu().detach().numpy()))))
#                 metrics['Saliency Var'].append(float(np.var(n_mask[top_prediction_class].cpu().detach().numpy())))
#                 metrics['L_Preserve Var'].append(np.var(metrics['L_Preserve']) if len(metrics['L_Preserve']) > 1 else 0.0)
#                 metrics['L_Budget Var'].append(np.var(metrics['L_Budget']) if len(metrics['L_Budget']) > 1 else 0.0)
#                 metrics['L_TV_Norm Var'].append(np.var(metrics['L_TV_Norm']) if len(metrics['L_TV_Norm']) > 1 else 0.0)
#                 metrics['L_SSD Var'].append(np.var(metrics['L_SSD']) if len(metrics['L_SSD']) > 1 else 0.0)
#                 metrics['L_Total Var'].append(np.var(metrics['L_Total']) if len(metrics['L_Total']) > 1 else 0.0)
#                 metrics['Confidence'].append(float(outputs[top_prediction_class][top_prediction_class].item()))

#                 if self.args.print == 'yes':
#                     m_data = (f'{i}/{self.args.max_itr}({i / self.args.max_itr * 100:.2f}%)'
#                             f' | P:{metrics["L_Preserve"][-1]: .4f}(V:{metrics["L_Preserve Var"][-1]:.4f})'
#                             # f' | B:{metrics["L_Budget"][-1]: .4f}(V:{metrics["L_Budget Var"][-1]: .4f})'
#                             # f' | TV:{metrics["L_TV_Norm"][-1]: .4f}(V: {metrics["L_TV_Norm Var"][-1]: .4f})'
#                             f' | S:{metrics["L_SSD"][-1]: .4f}(V:{metrics["L_SSD Var"][-1]: .4f})'
#                             f' | TL:{metrics["L_Total"][-1]: .4f}(V:{metrics["L_Total Var"][-1]: .4f})'
#                             f' | EPS:{self.eps:.4f} , M:{self.mode} | TCP:{outputs[top_prediction_class][top_prediction_class]: .4f} ({outputs[top_prediction_class][top_prediction_class]/top_prediction_confidence*100:.2f}%)')
#                     printr(m_data, sep='|')

#                 optimizer.step()

#                 self.rs_priority_buffer.update_priorities(rs_index, [rs_prios.item()])
#                 self.ro_priority_buffer.update_priorities(ro_index, [ro_prios.item()])

#                 if self.args.enable_lr_decay:
#                     scheduler.step(epoch=i)

#                 # Clamp mask
#                 n_mask.data.clamp_(-1, 1)

#                 if self.enable_wandb:
#                     _mets = {**{k: v[-1] for k, v in metrics.items() if k != "epoch"}}
#                     if f"epoch_{i}" in metrics["epoch"]:
#                         _mets = {**_mets, **metrics["epoch"][f"epoch_{i}"]['eval_metrics']}
#                     wandb.log(_mets)

#                 i += 1

#             # mask = n_mask[top_prediction_class].cpu().detach().numpy().flatten()
#             # if top_prediction_class == top_prediction_class2:
#             #     n_mask2.append(mask)
#             # else:
#             #     n_mask2.append(1-mask)

#             # mask = np.mean(np.array([mask_repo[m] for m in sorted(mask_repo)[0:10]]), axis=0)

#         # Fetch Top Class Mask
#         # mask = n_mask[top_prediction_class2].cpu().detach().numpy().flatten()
#         mask = n_mask[top_prediction_class].cpu().detach().numpy().flatten()

#         # Normalize Mask
#         norm_mask = (mask-mask.min())/(mask.max()-mask.min())

#         # return norm_mask, self.perturbation_manager, n_mask.cpu().detach().numpy().flatten()
#         return mask, self.perturbation_manager, n_mask.cpu().detach().numpy()
#         # return mask, self.perturbation_manager, n_mask2
