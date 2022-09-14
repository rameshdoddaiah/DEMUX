# -*- coding: utf-8 -*-
"""
| **@created on:** 8/30/20,
| **@author:** prathyushsp,
| **@version:** v0.0.1
|
| **Description:**
|
|
| **Sphinx Documentation Status:**
"""

from torch import nn
from tqdm import tqdm
from scipy.ndimage.filters import gaussian_filter
from sklearn.metrics import auc
from scipy.stats import skew, skewtest
import numpy as np
import torch
from matplotlib import pyplot as plt
import io
import wandb
from PIL import Image
import os
from nte.experiment.utils import send_plt_to_wandb
import multiprocessing
import platform
import scipy
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
import copy


def custom_auc(arr):
    """Returns normalized Area Under Curve of the array."""
    return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)


class CausalMetric():

    def __init__(self, model, mode, step, substrate_fn, supp=0.80):
        r"""Create deletion/insertion metric instance.
        Args:
            model (nn.Module): Black-box model being explained.
            mode (str): 'del' or 'ins'.
            step (int): number of pixels modified per one iteration.
            substrate_fn (func): a mapping from old pixels to new pixels.
        """
        assert mode in ['del', 'ins']
        self.model = model
        self.mode = mode
        self.step = step
        self.substrate_fn = substrate_fn
        self.ins_cutoff = 1.0
        self.del_cutoff = 0.0
        self.per_supp = supp
        self.per_supp_len = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        self.use_cuda = torch.cuda.is_available()
        self.device = 'cuda' if self.use_cuda else 'cpu'
        self.softmax = torch.nn.Softmax(dim=-1)
        # self.ins_cutoff = 0.99
        # self.del_cutoff = 0.10

    def single_run(self, dataset, args, class_0_mean, class_1_mean, time_series_tensor, explanation, enable_wandb, debug, return_results, save_to=None):
        TIMESTEPS = len(time_series_tensor)
        orig_pred, [orig_pred_class], orig_acts = self.model.evaluate(time_series_tensor.reshape(1, -1).to(self.device), args)

        salient_order = np.flip(np.argsort(explanation.reshape(-1, TIMESTEPS), axis=1), axis=-1)
        # Assign Start and Finish Tensors based on Insertion or Deletion
        if self.mode == 'del':
            title = 'Deletion Metric'
            ylabel = '% of Time Steps deleted'
            start = time_series_tensor.clone().to('cpu')
            if args.eval_replacement_mc == 'random_class_mean':
                rand_class_mean = getattr(dataset, f"test_class_{rand_class}_mean")
                finish = torch.tensor(rand_class_mean.reshape(1, -1), dtype=torch.float32).to('cpu')
            elif args.eval_replacement_mc == 'random_class_random_index':
                rand_class_indices = getattr(dataset, f"test_class_{rand_class}_indices")
                rand_ind = np.random.randint(0, len(rand_class_indices))
                rand_tclass_data = getattr(dataset, f"test_class_{rand_class}_data")
                finish = torch.tensor(rand_tclass_data[rand_ind].reshape(1, -1), dtype=torch.float32).to('cpu')
            elif args.eval_replacement_mc == 'max_class_random_index':
                max_class_indices = getattr(dataset, f"test_class_{max_class}_indices")
                rand_ind = np.random.randint(0, len(max_class_indices))
                max_tclass_data = getattr(dataset, f"test_class_{max_class}_data")
                finish = torch.tensor(max_tclass_data[rand_ind].reshape(1, -1), dtype=torch.float32).to('cpu')
            elif args.eval_replacement_mc == 'max_class_max_index':
                finish = time_series_tensor.clone()
            elif args.eval_replacement_mc == 'instance_mean':
                finish = np.mean(time_series_tensor.cpu().detach().numpy()).repeat(time_series_tensor.shape[0])
                finish = torch.tensor(finish.reshape(1, -1), dtype=torch.float32).to('cpu')
            elif args.eval_replacement_mc == 'min_class_random_index':
                min_class_indices = getattr(dataset, f"test_class_{min_class}_indices")
                rand_ind = np.random.randint(0, len(min_class_indices))
                min_tclass_data = getattr(dataset, f"test_class_{min_class}_data")
                finish = torch.tensor(min_tclass_data[rand_ind].reshape(1, -1), dtype=torch.float32).to('cpu')
            elif args.eval_replacement_mc == "cluster_mean":
                centroids = getattr(dataset, "cluster_meta")["centroids"]
                dist = [scipy.spatial.distance.euclidean(time_series_tensor.numpy().reshape(1, -1),
                                                         centroids[c]) for c in range(getattr(dataset, "cluster_meta")["n_clusters"])]
                finish = torch.tensor(centroids[np.argmax(dist)], dtype=torch.float32).to('cpu')
            elif args.eval_replacement_mc == 'min_class_mean':
                min_class_mean = getattr(dataset, f"test_class_{min_class}_mean")
                finish = torch.tensor(min_class_mean.reshape(1, -1), dtype=torch.float32).to('cpu')
            elif args.eval_replacement_mc == 'zeros':
                finish = torch.zeros_like(time_series_tensor)
            elif args.eval_replacement_mc == 'ones':
                finish = torch.ones_like(time_series_tensor)
            else:
                raise Exception("Unsupported eval replacement")
        elif self.mode == 'ins':
            title = 'Insertion Metric'
            ylabel = '% of Time steps inserted'
            finish = time_series_tensor.clone().to('cpu')
            if args.eval_replacement_mc == 'random_class_mean':
                rand_class_mean = getattr(dataset, f"test_class_{rand_class}_mean")
                start = torch.tensor(rand_class_mean.reshape(1, -1), dtype=torch.float32).to('cpu')

            elif args.eval_replacement_mc == 'rand_class_random_index':
                rand_class_indices = getattr(dataset, f"test_class_{rand_class}_indices")
                rand_ind = np.random.randint(0, len(rand_class_indices))
                rand_tclass_data = getattr(dataset, f"test_class_{rand_class}_data")
                start = torch.tensor(rand_tclass_data[rand_ind].reshape(1, -1), dtype=torch.float32).to('cpu')

            elif args.eval_replacement_mc == 'max_class_random_index':
                max_class_indices = getattr(dataset, f"test_class_{max_class}_indices")
                rand_ind = np.random.randint(0, len(max_class_indices))
                max_tclass_data = getattr(dataset, f"test_class_{max_class}_data")
                start = torch.tensor(max_tclass_data[rand_ind].reshape(1, -1), dtype=torch.float32).to('cpu')

            elif args.eval_replacement_mc == 'max_class_max_index':
                start = time_series_tensor.clone()

            elif args.eval_replacement_mc == 'instance_mean':
                start = np.mean(time_series_tensor.cpu().detach().numpy()).repeat(time_series_tensor.shape[0])
                start = torch.tensor(start.reshape(1, -1), dtype=torch.float32).to('cpu')

            elif args.eval_replacement_mc == 'min_class_random_index':
                min_class_indices = getattr(dataset, f"test_class_{min_class}_indices")
                rand_ind = np.random.randint(0, len(min_class_indices))
                min_tclass_data = getattr(dataset, f"test_class_{min_class}_data")
                start = torch.tensor(min_tclass_data[rand_ind].reshape(1, -1), dtype=torch.float32).to('cpu')

            elif args.eval_replacement_mc == "cluster_mean":
                centroids = getattr(dataset, "cluster_meta")["centroids"]
                dist = [scipy.spatial.distance.euclidean(time_series_tensor.numpy().reshape(1, -1),
                                                         centroids[c]) for c in range(getattr(dataset, "cluster_meta")["n_clusters"])]
                start = torch.tensor(centroids[np.argmax(dist)], dtype=torch.float32).to('cpu')

            elif args.eval_replacement_mc == 'min_class_mean':
                min_class_mean = getattr(dataset, f"test_class_{min_class}_mean")
                start = torch.tensor(min_class_mean.reshape(1, -1), dtype=torch.float32).to('cpu')

            elif args.eval_replacement_mc == 'zeros':
                start = torch.zeros_like(time_series_tensor)
            elif args.eval_replacement_mc == 'ones':
                start = torch.ones_like(time_series_tensor)
            else:
                raise Exception("Unsupported eval replacement")
        else:
            raise Exception('error in mode')

        n_steps = (TIMESTEPS + self.step - 1) // self.step
        scores = np.zeros(int(n_steps) + 1)
        for i in range(n_steps + 1):
            pred, cl, acts = self.model.evaluate(start.reshape(1, -1).to(self.device), args)
            scores[i] = 1-(abs(orig_pred-acts[0][orig_pred_class].item())/orig_pred)
            if i < n_steps:
                coords = salient_order[:, self.step * i:self.step * (i + 1)][0]
                start.cpu().numpy().reshape(TIMESTEPS)[coords] = finish.cpu().numpy().reshape(TIMESTEPS)[coords]
        return_results[self.mode] = [scores, [], 0, 0]
        return return_results

    def single_run_v0(self, dataset, args, class_0_mean, class_1_mean, time_series_tensor, explanation, enable_wandb, debug,
                      return_results, save_to=None):
                #    return_results, save_to="./wandb/"):
        r"""Run metric on one image-saliency pair.
        Args:
            img_tensor (Tensor): normalized image tensor.
            explanation (np.ndarray): saliency map.
            verbose (int): in [0, 1, 2].
                0 - return list of scores.
                1 - also plot final step.
                2 - also plot every step and print 2 top classes.
            save_to (str): directory to save every step plots to.
        Return:
            scores (nd.array): Array containing scores at every step.
        """
        time_series_tensor = time_series_tensor.to(self.device)
        X = time_series_tensor.clone()
        TIMESTEPS = len(time_series_tensor)
        # todo Ramesh: Check whether the model give raw prediction?
        pred, c, acts = self.model.evaluate(time_series_tensor.reshape(1, -1).to(self.device), args)
        orig_pred = pred
        c = c[0]
        n_steps = (TIMESTEPS + self.step - 1) // self.step
        # cla = np.zeros(int(n_steps) + 1)
        # Coordinates of pixels in order of decreasing saliency
        salient_order = np.flip(np.argsort(explanation.reshape(-1, TIMESTEPS), axis=1), axis=-1)
        # print("Salient Order keys", salient_order)
        # print("Salient order values", explanation[salient_order])
        # print("Explanation", explanation)
        # print("percentile", np.percentile(explanation, 50, axis=1,keepdims=True))
        orig_cl = c.item()
        ret_auc = 0.0

        # print('Original Prediction: ', pred.item(), "class", c.item())
        # print(f'orig Pre{ pred.item()} clas {c.item()} acts {acts}')

        sal_order = {}

        t_acts = torch.tensor(acts)
        t_values, t_indices = t_acts.topk(dataset.num_classes)
        if args.second_class == "yes":
            orig_cl = t_indices[0][1]
            # print(f"orig_cl second_class {orig_cl} pred {t_values[0][1]}")
        elif args.third_class == "yes":
            orig_cl = t_indices[0][2]
            # print(f"orig_cl third_class {orig_cl} pred {t_values[0][1]}")
        else:
            # print(f"orig_cl first class{orig_cl} pred {t_values[0][1]}")
            orig_cl = orig_cl
        # print(f"t values {t_values} t indices {t_indices} t_acts {t_acts}")
        rand_choice = list(range(0, orig_cl)) + list(range(orig_cl+1, dataset.num_classes))
        rand_class = np.random.choice(rand_choice)
        # min_class = np.argmin(t_values).item()
        # max_class = np.argmax(t_values).item()
        min_class = np.argmin(acts)
        max_class = np.argmax(acts)
        print(f"min_class {min_class} max_class{max_class} acts {acts} ")
        # rand_class = min_class
        # rand_class = max_class
        t_acts = torch.tensor(acts)
        t_values, t_indices = t_acts.topk(dataset.num_classes)
        # rand_class = t_indices[0][dataset.num_classes-1]
        # rand_class = t_indices[0][0]

        if self.mode == 'del':
            title = 'Deletion Metric'
            ylabel = '% of Time Steps deleted'
            if args.second_class == "yes":
                orig_cl_mean = getattr(dataset, f"test_class_{orig_cl}_mean")
                start = torch.tensor(orig_cl_mean.reshape(1, -1), dtype=torch.float32).to('cpu')
            elif args.third_class == "yes":
                orig_cl_mean = getattr(dataset, f"test_class_{orig_cl}_mean")
                start = torch.tensor(orig_cl_mean.reshape(1, -1), dtype=torch.float32).to('cpu')
            else:
                start = time_series_tensor.clone().to('cpu')
            if args.multi_class == 'yes':
                # rand_class_mean = getattr(dataset, f"test_class_{rand_class}_mean")
                # finish = torch.tensor(rand_class_mean.reshape(1, -1), dtype=torch.float32).to('cpu')
                if args.eval_replacement_mc == 'random_class_mean':
                    rand_class_mean = getattr(dataset, f"test_class_{rand_class}_mean")
                    finish = torch.tensor(rand_class_mean.reshape(1, -1), dtype=torch.float32).to('cpu')

                elif args.eval_replacement_mc == 'random_class_random_index':
                    rand_class_indices = getattr(dataset, f"test_class_{rand_class}_indices")
                    rand_ind = np.random.randint(0, len(rand_class_indices))
                    rand_tclass_data = getattr(dataset, f"test_class_{rand_class}_data")
                    finish = torch.tensor(rand_tclass_data[rand_ind].reshape(1, -1), dtype=torch.float32).to('cpu')

                elif args.eval_replacement_mc == 'max_class_random_index':
                    max_class_indices = getattr(dataset, f"test_class_{max_class}_indices")
                    rand_ind = np.random.randint(0, len(max_class_indices))
                    max_tclass_data = getattr(dataset, f"test_class_{max_class}_data")
                    finish = torch.tensor(max_tclass_data[rand_ind].reshape(1, -1), dtype=torch.float32).to('cpu')

                elif args.eval_replacement_mc == 'max_class_max_index':
                    finish = time_series_tensor.clone()

                elif args.eval_replacement_mc == 'instance_mean':
                    finish = np.mean(time_series_tensor.cpu().detach().numpy()).repeat(time_series_tensor.shape[0])
                    finish = torch.tensor(finish.reshape(1, -1), dtype=torch.float32).to('cpu')

                elif args.eval_replacement_mc == 'min_class_random_index':
                    min_class_indices = getattr(dataset, f"test_class_{min_class}_indices")
                    rand_ind = np.random.randint(0, len(min_class_indices))
                    min_tclass_data = getattr(dataset, f"test_class_{min_class}_data")
                    finish = torch.tensor(min_tclass_data[rand_ind].reshape(1, -1), dtype=torch.float32).to('cpu')
                elif args.eval_replacement_mc == 'min_class_mean':
                    min_class_mean = getattr(dataset, f"test_class_{min_class}_mean")
                    finish = torch.tensor(min_class_mean.reshape(1, -1), dtype=torch.float32).to('cpu')

                elif args.eval_replacement_mc == 'zeros':
                    finish = torch.zeros_like(time_series_tensor)
                elif args.eval_replacement_mc == 'ones':
                    finish = torch.ones_like(time_series_tensor)
                else:
                    raise Exception("Unsupported eval replacement")
            else:
                if orig_cl == 0:
                    finish = torch.tensor(class_1_mean.reshape(1, -1), dtype=torch.float32).to('cpu')
                else:
                    finish = torch.tensor(class_0_mean.reshape(1, -1), dtype=torch.float32).to('cpu')
        elif self.mode == 'ins':
            title = 'Insertion Metric'
            ylabel = '% of Time steps inserted'
            if args.multi_class == 'yes':
                if args.eval_replacement_mc == 'random_class_mean':
                    rand_class_mean = getattr(dataset, f"test_class_{rand_class}_mean")
                    start = torch.tensor(rand_class_mean.reshape(1, -1), dtype=torch.float32).to('cpu')

                elif args.eval_replacement_mc == 'rand_class_random_index':
                    rand_class_indices = getattr(dataset, f"test_class_{rand_class}_indices")
                    rand_ind = np.random.randint(0, len(rand_class_indices))
                    rand_tclass_data = getattr(dataset, f"test_class_{rand_class}_data")
                    start = torch.tensor(rand_tclass_data[rand_ind].reshape(1, -1), dtype=torch.float32).to('cpu')

                elif args.eval_replacement_mc == 'max_class_random_index':
                    max_class_indices = getattr(dataset, f"test_class_{max_class}_indices")
                    rand_ind = np.random.randint(0, len(max_class_indices))
                    max_tclass_data = getattr(dataset, f"test_class_{max_class}_data")
                    start = torch.tensor(max_tclass_data[rand_ind].reshape(1, -1), dtype=torch.float32).to('cpu')

                elif args.eval_replacement_mc == 'max_class_max_index':
                    start = time_series_tensor.clone()

                elif args.eval_replacement_mc == 'instance_mean':
                    start = np.mean(time_series_tensor.cpu().detach().numpy()).repeat(time_series_tensor.shape[0])
                    start = torch.tensor(start.reshape(1, -1), dtype=torch.float32).to('cpu')

                elif args.eval_replacement_mc == 'min_class_random_index':
                    min_class_indices = getattr(dataset, f"test_class_{min_class}_indices")
                    rand_ind = np.random.randint(0, len(min_class_indices))
                    min_tclass_data = getattr(dataset, f"test_class_{min_class}_data")
                    start = torch.tensor(min_tclass_data[rand_ind].reshape(1, -1), dtype=torch.float32).to('cpu')

                elif args.eval_replacement_mc == 'min_class_mean':
                    min_class_mean = getattr(dataset, f"test_class_{min_class}_mean")
                    start = torch.tensor(min_class_mean.reshape(1, -1), dtype=torch.float32).to('cpu')

                elif args.eval_replacement_mc == 'zeros':
                    start = torch.zeros_like(time_series_tensor)
                elif args.eval_replacement_mc == 'ones':
                    start = torch.ones_like(time_series_tensor)
                else:
                    raise Exception("Unsupported eval replacement")

                # rand_class_indices = getattr(dataset, f"train_class_{max_class}_indices")
                # rand_ind = np.random.randint(0,len(rand_class_indices))
                # rand_tclass_data = getattr(dataset, f"train_class_{max_class}_data")
                # start = torch.tensor(rand_tclass_data[rand_ind].reshape(1, -1), dtype=torch.float32).to('cpu')

                # start = torch.tensor(rand_tclass_data[rand_class_indices[0]].reshape(1, -1), dtype=torch.float32).to('cpu')
                # start = time_series_tensor.clone().to('cpu')
                # ins_mean = np.mean(time_series_tensor.cpu().detach().numpy()).repeat(time_series_tensor.shape[0])
                # start = torch.tensor(ins_mean.reshape(1, -1), dtype=torch.float32).to('cpu')
            else:
                if orig_cl == 0:
                    start = torch.tensor(class_1_mean.reshape(1, -1), dtype=torch.float32).to('cpu')
                else:
                    start = torch.tensor(class_0_mean.reshape(1, -1), dtype=torch.float32).to('cpu')
            if args.second_class == "yes":
                orig_cl_mean = getattr(dataset, f"test_class_{orig_cl}_mean")
                finish = torch.tensor(orig_cl_mean.reshape(1, -1), dtype=torch.float32).to('cpu')
            elif args.third_class == "yes":
                orig_cl_mean = getattr(dataset, f"test_class_{orig_cl}_mean")
                finish = torch.tensor(orig_cl_mean.reshape(1, -1), dtype=torch.float32).to('cpu')
            else:
                finish = time_series_tensor.clone().to('cpu')
        else:
            raise Exception('error in mode')

        scores = np.zeros(int(n_steps) + 1)

        # minimality of saliency
        sal = explanation.flatten()
        bins = {b: [] for b in np.linspace(0, 1, 225)}
        for s in sal:
            for b in bins.keys():
                if s <= b:
                    bins[b].append(s)
                    break
        vs = [len(b) / len(sal) for b in bins.values()]
        new_cl = []

        CRED = '\033[91m'
        CEND = '\033[0m'

        CGRN = '\033[92m'
        # CGRN = '\033[0m'

        for i in range(n_steps + 1):
            # if self.mode == 'del':
            #     print(CRED+f"{self.mode} -  Step: {i}/{n_steps + 1} ({i / (n_steps + 1) * 100:.2f}%) \n Start {start} Saliency {salient_order} \n Explanation {explanation} \n Original time Series {time_series_tensor} finish{finish}"+CEND)
            # else:
            #     print(CGRN+f"{self.mode} -  Step: {i}/{n_steps + 1} ({i / (n_steps + 1) * 100:.2f}%) \n Start {start} Saliency {salient_order} \n Explanation {explanation} \n Original time Series {time_series_tensor} finish{finish}"+CEND)
            # r7878
            pred, cl, acts = self.model.evaluate(start.reshape(1, -1).to(self.device), args)
            scores[i] = acts[0][orig_cl].item()
            new_cl.append(cl.item())
            # if self.mode == 'del':
            #     if args.cricketx == 'yes':
            #         rand_class = np.argmin(acts)
            #         rand_class_mean = getattr(dataset, f"test_class_{rand_class}_mean")
            #         finish = torch.tensor(rand_class_mean.reshape(1, -1), dtype=torch.float32).to('cpu')
            # else:
            #     if args.cricketx == 'yes':
            #         rand_class = np.argmin(acts)
            #         rand_class_mean = getattr(dataset, f"test_class_{rand_class}_mean")
            #         start = torch.tensor(rand_class_mean.reshape(1, -1), dtype=torch.float32).to('cpu')

            # cla[i] = cl.item()
            if self.mode == 'del':
                if (scores[i] >= 0.95):
                    self.per_supp_len[0] = i + 1
                if (scores[i] >= 0.90):
                    self.per_supp_len[1] = i + 1
                if (scores[i] >= 0.80):
                    self.per_supp_len[2] = i + 1
                if (scores[i] >= 0.70):
                    self.per_supp_len[3] = i + 1
                if (scores[i] >= 0.60):
                    self.per_supp_len[4] = i + 1
                if (scores[i] >= 0.50):
                    self.per_supp_len[5] = i + 1
                if (scores[i] >= 0.40):
                    self.per_supp_len[6] = i + 1
                if (scores[i] >= 0.30):
                    self.per_supp_len[7] = i + 1
                if (scores[i] >= 0.20):
                    self.per_supp_len[8] = i + 1
                if (scores[i] >= 0.10):
                    self.per_supp_len[9] = i + 1

            if debug:
                print(
                    f"Step:{i} mode: {self.mode} Pred:{pred}  Orig_pred:{orig_pred}  Cl:{cl}  Orig_cl:{orig_cl}  Scores:{scores[i]}")
                print(f"Acts:{acts}")
                # print(f"t_values:{t_values}")
                # print(f"t_indices:{t_indices}")

            # scores[i] = pred[c]

            if debug or save_to:
                # if(len(np.arange(i+1)) != len(vs[:i+1])):
                #     continue
                plt.figure(figsize=(10, 5))
                plt.subplot(131)
                plt.title('{} {:.1f}%, P(1)={:.2f}'.format(ylabel, 100 * (i - 1) / n_steps, scores[i]))
                plt.plot(list(range(len(X))), X, label="Raw Pattern", color="red", alpha=0.4)
                # plt.plot(list(range(len(start))), start,label="Perturbed Pattern")
                plt.scatter(list(sal_order.keys()), list(sal_order.values()), color="orange", label="Saliency")
                plt.xlabel("Timesteps")
                plt.ylabel("Values")
                plt.legend()

                plt.subplot(132)
                plt.plot(np.arange(i + 1) / n_steps, scores[:i + 1], label="AUC")
                plt.xlim(-0.1, 1.1)
                plt.ylim(0, 1.05)
                plt.fill_between(np.arange(i + 1) / n_steps, 0, scores[:i + 1], alpha=0.4)
                plt.title(title + f" AUC - {auc(np.arange(i + 1) / n_steps, scores[:i + 1]) if i > 5 else 0.0:.4f}")
                plt.xlabel(ylabel)
                plt.ylabel("Probability")
                plt.title(title + f" AUC - {auc(np.arange(i + 1) / n_steps, scores[:i + 1]) if i > 5 else 0.0:.4f}")
                # plt.text(x=0.4, y=0.95, s=f'AUC: {auc(scores[:i + 1]):.2f}')
                plt.legend()

                plt.subplot(133)
                plt.plot(np.arange(i + 1) / n_steps, vs[:i + 1])
                plt.fill_between(list(bins.keys())[:i + 1], 0, vs[:i + 1], alpha=0.4)
                plt.yscale("log")
                plt.xlabel("% of Time Steps")
                plt.ylabel("Saliency")
                plt.title(f"% Saliency AUC - {auc(list(bins.keys())[:i + 1], vs[:i + 1]) if i > 5 else 0.0:.4f}")
                if save_to:
                    plt.tight_layout()
                    plt.savefig(save_to + '/{:03d}.png'.format(i))
                    if enable_wandb:
                        wandb.log({f"Evaluation  - {title}": plt})
                        wandb.log({f"{title} scores": float(scores[i]),
                                   f"{title} scores ms": float(vs[i])})
                    plt.close()
                else:
                    plt.show()
            if i < n_steps:
                coords = salient_order[:, self.step * i:self.step * (i + 1)][0]
                # if self.mode == 'del':
                #     print(CRED+f"1:coords{coords} scores {scores}, start {start.cpu().numpy().reshape(TIMESTEPS)[coords]} finish {finish.cpu().numpy().reshape(TIMESTEPS)[coords]}"+CEND)
                # else:
                #     print(CGRN+f"1:coords{coords} scores{scores}, start {start.cpu().numpy().reshape(TIMESTEPS)[coords]} finish {finish.cpu().numpy().reshape(TIMESTEPS)[coords]}"+CEND)
                sal_order[coords[0]] = explanation[int(coords[0])]
                start.cpu().numpy().reshape(TIMESTEPS)[coords] = finish.cpu().numpy().reshape(TIMESTEPS)[coords]
                # if self.mode == 'del':
                #     print(CRED+f"2:coords{coords} scores{scores}, start {start.cpu().numpy().reshape(TIMESTEPS)[coords]} finish {finish.cpu().numpy().reshape(TIMESTEPS)[coords]}"+CEND)
                # else:
                #     print(CGRN+f"2:coords{coords} scores{scores}, start {start.cpu().numpy().reshape(TIMESTEPS)[coords]} finish {finish.cpu().numpy().reshape(TIMESTEPS)[coords]}"+CEND)

        if self.mode == 'del':
            # deletion game metrics
            print('Percent Suppression Length : ', self.per_supp_len)
            plt.figure()
            plt.title("Percent Suppression Score")
            # for x in range(1,10,1):
            #     plt.bar(str(x*10)+"%", height=self.per_supp_len[x], color=np.random.rand(3), align='center')
            plt.bar("10%", height=self.per_supp_len[0], color=np.random.rand(3), align='center')
            plt.bar("20%", height=self.per_supp_len[1], color=np.random.rand(3), align='center')
            plt.bar("30%", height=self.per_supp_len[2], color=np.random.rand(3), align='center')
            plt.bar("40%", height=self.per_supp_len[3], color=np.random.rand(3), align='center')
            plt.bar("50%", height=self.per_supp_len[4], color=np.random.rand(3), align='center')
            plt.bar("60%", height=self.per_supp_len[5], color=np.random.rand(3), align='center')
            plt.bar("70%", height=self.per_supp_len[6], color=np.random.rand(3), align='center')
            plt.bar("80%", height=self.per_supp_len[7], color=np.random.rand(3), align='center')
            plt.bar("90%", height=self.per_supp_len[8], color=np.random.rand(3), align='center')
            plt.bar("95%", height=self.per_supp_len[9], color=np.random.rand(3), align='center')
            # plt.bar(1,height=self.per_supp_len,color='b',width60.1)
            plt.xlabel("Percent of model output suppression")
            plt.ylabel("TimeSteps")
            plt.legend()
            if save_to:
                plt.savefig(save_to + '_supp.png')
                if enable_wandb:
                    wandb.log({f"Deletion Game": [send_plt_to_wandb(plt, 'Deletion Game')]})
                plt.close()
            else:
                plt.show()
        return_results[self.mode] = [scores, np.array(vs), ret_auc, self.per_supp_len]
        return return_results

def compute_saliency_distance(X, model, saliency_t, n_masks_t, classes, top_class):
    # Euclidean distance
    # DTW distance
    # Frechet distance for MultiVariate
    saliency = copy.deepcopy(saliency_t)
    n_masks = copy.deepcopy(n_masks_t)
    saliency[saliency <=0] = 0

    for i in range(classes):
        n_masks[i][n_masks[i] <=0] = 0

    cos_distance = 0
    euc_distance = 0
    for i in range(classes):
        if i != top_class:
            if(np.isnan(saliency).any()):
                print("Nan values saliency")
                print(saliency)
                print(n_masks[i])
                continue

            if(np.isnan(n_masks[i]).any()):
                print("Nan values n_masks")
                print(saliency)
                print(n_masks[i])
                continue

            cos_distance+= cosine_similarity([saliency],[n_masks[i]])[0][0]
            euc_distance+= euclidean_distances([saliency],[n_masks[i]])[0][0]

    cos_similar_distance = (cos_distance/((classes-1)))
    euc_dissimilar_distance = (euc_distance/((classes-1)))

    return cos_similar_distance, euc_dissimilar_distance


def compute_iou_auc(saliency, n_masks, classes, top_class):
    alphas = [i/10 for i in range(1, 10)]
    ious = []
    saliency = np.array(saliency)
    n_masks = np.array(n_masks)
    for alpha in alphas:
        temp_saliency = copy.deepcopy(saliency)
        temp_n_masks = copy.deepcopy(n_masks)
        temp_saliency[saliency >= alpha] = 1
        temp_saliency[saliency < alpha] = 0
        temp_saliency = temp_saliency.astype(int)

        for i in range(len(temp_n_masks)):
            temp_n_masks[i][temp_n_masks[i] >= alpha] = 1
            temp_n_masks[i][temp_n_masks[i] < alpha] = 0

        temp_n_masks = np.array(temp_n_masks)
        temp_n_masks = temp_n_masks.astype(int)

        iou_mean = 0
        for i in range(classes):
            if i != top_class:
                intersection = np.count_nonzero(temp_saliency & temp_n_masks[i]) + 1e-5
                union = np.count_nonzero(temp_saliency | temp_n_masks[i]) + 1e-5
                iou_mean+=(intersection/union)
        iou_mean = iou_mean/(classes-1)
        ious.append(iou_mean)
    return ious, auc(x=list(range(len(ious))), y=ious) / (len(ious) - 1)


def compute_iou(X, model, saliency_t, n_masks_t, classes, top_class, main_args):
    saliency = copy.deepcopy(saliency_t)
    n_masks = copy.deepcopy(n_masks_t)
    alpha = main_args.iou_alpha
    saliency[saliency >=alpha] = 1
    saliency[saliency <alpha] = 0

    for i in range(classes):
        if(np.isnan(n_masks[i]).any()):
            continue
        n_masks[i][n_masks[i] >=alpha] = 1
        n_masks[i][n_masks[i] <alpha] = 0
        n_masks[i] = n_masks[i].astype(int)

    intersection = 0
    union = np.zeros_like(saliency).astype(int)
    iou_avg = 0

    for i in range(classes):
        if i != top_class:
            if(np.isnan(n_masks[i]).any()):
                iou_avg+=1
                continue
            # intersection = (intersection.astype(int)) & ( n_masks[i].astype(int) & n_masks[top_class].astype(int))
            intersection=np.count_nonzero(n_masks[i].astype(int) & n_masks[top_class].astype(int))
            union = (union.astype(int)) | (n_masks[i].astype(int) | n_masks[top_class].astype(int))
            union_count = np.count_nonzero(union == 1)
            if union_count  <= 0 :
                continue
            union_count = saliency_t.shape[0]
            iou_avg+=intersection/(union_count)

    # ts_len = saliency_t.shape[0]
    # union_count = np.count_nonzero(union == 1)
    # if intersection > ts_len or union_count > ts_len or union_count  == 0 :
    #     return 0
    # iou_metrics = intersection/union_count
    iou_metrics = iou_avg/(classes-1)

    return iou_metrics,

def qm_plot(X, model, saliency, n_masks, classes, top_class, class_0_mean, class_1_mean, enable_wandb, save_dir, supp=0.08, debug=False, multi_process=True, dataset=None, main_args=None):
    if 'Darwin' in platform.platform():
        multiprocessing.set_start_method('forkserver', force=True)
        multi_process = True
    manager = multiprocessing.Manager()

    process_results = manager.dict()
    TIMESTEPS = len(X)
    STEPS = 20
    # orig STEPS = 20
    # STEPS = 1
    # noise = torch.tensor(generate_gaussian_noise(X, snrdb=0.001), dtype=torch.float32)
    # noise = torch.tensor(np.random.random(size=TIMESTEPS), dtype=torch.float32)

    # noise = 0.01
    # def blur(x): return x * noise
    eval_metrics = {}
    insertion = CausalMetric(model, 'ins', STEPS, substrate_fn=torch.zeros_like, supp=supp)
    deletion = CausalMetric(model, 'del', STEPS, substrate_fn=torch.zeros_like, supp=supp)

    if multi_process:
        p1 = multiprocessing.Process(target=insertion.single_run, args=(dataset, main_args, class_0_mean, class_1_mean,
                                                                        torch.tensor(X, dtype=torch.float32),
                                                                        saliency, enable_wandb, debug, process_results))
        p2 = multiprocessing.Process(target=deletion.single_run, args=(dataset, main_args, class_0_mean, class_1_mean,
                                                                       torch.tensor(X, dtype=torch.float32),
                                                                       saliency, enable_wandb, debug, process_results))
        p1.start()
        p2.start()
        p1.join()
        ins_scores, ins_ms, ins_ret_auc, ins_ips_len = process_results['ins']
    else:
        ins_scores, ins_ms, ins_ret_auc, ins_ips_len = insertion.single_run(dataset, main_args, class_0_mean, class_1_mean,
                                                                            torch.tensor(X, dtype=torch.float32),
                                                                            saliency, enable_wandb, debug, process_results)['ins']

    ins_trap_auc = auc(x=list(range(len(ins_scores))), y=ins_scores) / ((len(ins_scores) - 1))
    ins_cauc = custom_auc(ins_scores)
    print(f'Insertion - TrapAUC: {ins_trap_auc: .2f} | DiagAUC: {ins_cauc: .2f}')

    # trapauc_ms = auc(x=list(range(len(ins_ms))), y=ins_ms) / ((len(ins_ms) - 1))
    # cauc_ms = custom_auc(ins_ms)
    # stest = skewtest(ins_scores)
    # stest_ms = skewtest(ins_ms)
    eval_metrics['Insertion'] = {"trap": ins_trap_auc,
                                 "custom": ins_cauc,
                                #  "skew": skew(ins_scores),
                                 #  "pval": float(stest.pvalue),
                                 #  "zscore": float(stest.statistic),
                                 "scores": ins_scores.tolist(),
                                #  "trap_ms": trapauc_ms,
                                #  "custom_ms": cauc_ms,
                                 #  "skew_ms": skew(ins_ms),
                                 #  "pval_ms": float(stest_ms.pvalue),
                                 #  "zscore_ms": float(stest_ms.statistic),
                                 #  "scores_ms": ins_ms.tolist(),
                                 }

    if multi_process:
        p2.join()
        del_scores, del_ms, del_ret_auc, del_dps_len = process_results['del']
    else:
        del_scores, del_ms, del_ret_auc, del_dps_len = deletion.single_run(dataset, main_args, class_0_mean, class_1_mean,
                                                               torch.tensor(X, dtype=torch.float32),
                                                               saliency, enable_wandb, debug, process_results)['del']
    del_trapauc = auc(x=list(range(len(del_scores))), y=del_scores) / ((len(del_scores) - 1))
    del_cauc = custom_auc(del_scores)

    iou = 0
    cosine_sim = 0
    euc_dist = 0

    # ious,iou_auc=compute_iou_auc(X, model, saliency, n_masks, classes, top_class,main_args)
    ious, iou_auc = compute_iou_auc(saliency, n_masks, classes, top_class)
    cosine_sim, euc_dist = compute_saliency_distance(X, model, saliency, n_masks, classes, top_class)
    # trapauc_ms = auc(x=list(range(len(del_ms))), y=del_ms) / ((len(del_ms) - 1))
    # cauc_ms = custom_auc(del_ms)

    # stest = skewtest(del_scores)
    # stest_ms = skewtest(del_ms)

    alpha = main_args.iou_alpha
    bsaliency = copy.deepcopy(saliency)
    bsaliency[bsaliency >=alpha] = 1
    bsaliency[bsaliency <alpha] = 0

    eval_metrics['Deletion'] = {"trap": del_trapauc,
                                "custom": del_cauc,
                                # "skew": skew(del_scores),
                                # "pval": float(stest.pvalue),
                                # "zscore": float(stest.statistic),
                                "scores": del_scores.tolist(),
                                # "trap_ms": trapauc_ms,
                                # "custom_ms": cauc_ms,
                                # "skew_ms": skew(del_ms),
                                # "pval_ms": float(stest_ms.pvalue),
                                # "zscore_ms": float(stest_ms.statistic),
                                # "scores_ms": del_ms.tolist(),
                                }
    eval_metrics['Final'] = {'AUC': eval_metrics['Insertion']['trap'] - eval_metrics['Deletion']['trap'],
                             'AUC_difference': float(ins_trap_auc-del_trapauc),
                             "ms_sum_b": float(np.count_nonzero(bsaliency == 1)),
                             "ms_var_b": float(np.var(saliency > 0)),
                             "ms_sum_pos": float(np.sum(saliency>0)),
                             "ms_sum": float(np.sum(saliency)),
                             "ms_var": float(np.var(saliency)),
                             "cosine_sim":float(cosine_sim),
                             "euc_dist":float(euc_dist),
                             "iou_auc": float(iou_auc)
                            #  "iou_metrics":float(iou)
                             }
    # eval_metrics["ms_sum"]= np.sum(saliency)
    # eval_metrics["ms_var"]= np.var(saliency)

    # percs = ["10", "20", "30", "40", "50", "60", "70", "80", "90", "95"]
    # eval_metrics['Deletion Game'] = {k: v for k, v in zip(percs, del_dps_len)}

    print(f'Deletion - TrapAUC: {del_trapauc: .2f} | DiagAUC: {del_cauc: .2f}')
    print(f"AUC: {eval_metrics['Final']['AUC']:.2f} | IOU: {iou_auc:.2f}")
    print(eval_metrics['Final'])
    return eval_metrics


# ["zeros", "class_mean", "instance_mean", "random_instance", "random_opposing_instance"])
def run_evaluation_metrics(EVAL_REPLACEMENT, dataset, original_signal, model, mask, n_masks, classes, top_class, SAVE_DIR, ENABLE_WANDB, multi_process=False, main_args=None):
    if EVAL_REPLACEMENT == 'zeros':
        class_0_mean = np.repeat(0, len(dataset.test_class_0_mean))
        class_1_mean = np.repeat(0, len(dataset.test_class_1_mean))
    elif EVAL_REPLACEMENT == 'class_mean':
        class_0_mean = dataset.test_class_0_mean
        class_1_mean = dataset.test_class_1_mean
    elif EVAL_REPLACEMENT == 'instance_mean':
        ins_mean = np.mean(original_signal.cpu().detach().numpy()).repeat(
            original_signal.shape[0])
        class_0_mean = ins_mean
        class_1_mean = ins_mean
    else:
        raise Exception("Unsupported eval replacement")
    return qm_plot(model=model, X=original_signal, saliency=mask, n_masks=n_masks,classes=classes,top_class= top_class,
                   class_0_mean=class_0_mean,
                   class_1_mean=class_1_mean,
                   save_dir=SAVE_DIR, enable_wandb=ENABLE_WANDB, multi_process=multi_process, dataset=dataset, main_args=main_args)
