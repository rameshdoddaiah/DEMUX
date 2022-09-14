#Metrics Start
import numpy as np
import torch
from scipy.ndimage.filters import gaussian_filter
import torch.nn as nn
from scipy.ndimage.filters import gaussian_filter
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import matplotlib
from torch import nn
from tqdm import tqdm
from scipy.ndimage.filters import gaussian_filter
import torch.nn as nn
from matplotlib import pyplot as plt
from nte.models import BlipV3DNNModel, BlipV3RNNModel
import seaborn as sns

sns.set_style("darkgrid")
matplotlib.rc('xtick', labelsize=18)
matplotlib.rc('ytick', labelsize=18)
matplotlib.rc('axes', labelsize=18)
matplotlib.rc('axes', titlesize=18)
matplotlib.rc('legend', fontsize=18)
matplotlib.rc('figure', titlesize=18)
matplotlib.rc('figure', titlesize=18)

def gkern(klen, nsig):
    """Returns a Gaussian kernel array.
    Convolution with it results in image blurring."""
    # create nxn zeros
    inp = np.zeros((klen, klen))
    # set element at the middle to one, a dirac delta
    inp[klen//2, klen//2] = 1
    # gaussian-smooth the dirac, resulting in a gaussian filter mask
    k = gaussian_filter(inp, nsig)
    kern = np.zeros((3, 3, klen, klen))
    kern[0, 0] = k
    kern[1, 1] = k
    kern[2, 2] = k
    return torch.from_numpy(kern.astype('float32'))

klen = 11
ksig = 5
kern = gkern(klen, ksig)

# Function that blurs input image

# def blur(x): return nn.functional.conv1d(x, kern, padding=klen//2)

def generate_gaussian_noise(data, snrdb: float = 20.0):
    # Set a target SNR
    target_snr_db = snrdb
    # Calculate signal power and convert to dB
    x_watts = data ** 2
    sig_avg_watts = np.mean(x_watts)
    sig_avg_db = 10 * np.log10(sig_avg_watts)
    # Calculate noise according to [2] then convert to watts
    noise_avg_db = sig_avg_db - target_snr_db
    noise_avg_watts = 10 ** (noise_avg_db / 10)
    # Generate an sample of white noise
    mean_noise = 0
    noise_volts = np.random.normal(
        mean_noise, np.sqrt(noise_avg_watts), size=x_watts.shape)
# Noise up the original signal
    return noise_volts


# TIMESTEPS = 10
# n_classes = 2

def auc(arr):
    """Returns normalized Area Under Curve of the array."""
    return (arr.sum() - arr[0] / 2 - arr[-1] / 2) / (arr.shape[0] - 1)

class CausalMetric():

    def __init__(self, model, mode, step, substrate_fn):
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
        self.ins_cutoff = 0.95
        self.del_cutoff = 0.10

    def single_run(self, time_series_tensor, explanation, verbose=0, save_to=None):
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
        # pred = self.model(time_series_tensor)
        # top, c = torch.max(pred, 0)
        # c = c.cpu().numpy()

        TIMESTEPS = len(X)
        pred, cl, acts = model.evaluate(torch.tensor(
            time_series_tensor.reshape(1, -1), dtype=torch.float32).to(device))

        top = acts[0][cl]
        c = cl
        c = c
        n_steps = (TIMESTEPS + self.step - 1) // self.step
        n_steps = 15

        sal_order = {}

        if self.mode == 'del':
            title = 'Deletion Metric'
            ylabel = '% of Time Steps deleted'
            start = time_series_tensor.clone()
            finish = self.substrate_fn(time_series_tensor)
        elif self.mode == 'ins':
            title = 'Insertion Metric'
            ylabel = '% of Time steps inserted'
            start = self.substrate_fn(time_series_tensor)
            finish = time_series_tensor.clone()
        else:
            raise Exception('error in mode')

        scores = np.zeros(int(n_steps) + 1)
        # Coordinates of pixels in order of decreasing saliency
        salient_order = np.flip(np.argsort(
            explanation.reshape(-1, TIMESTEPS), axis=1), axis=-1)
        print("Salient Order keys", salient_order)
        print("Salient order values", explanation[salient_order])
        print("Explanation", explanation)
        # print("percentile", np.percentile(explanation, 50, axis=1,keepdims=True))
        pred, cl, acts = model.evaluate(torch.tensor(
            time_series_tensor.reshape(1, -1), dtype=torch.float32).to(device))

        print('Original Prediction: ', pred.item(), cl.item())
        # pred, cl, acts = trained_model.evaluate(torch.tensor(X.reshape(1, -1), dtype=torch.float32).to(device))
        for i in range(n_steps+1):
            # pred= self.model(start)
            # pred, cl, acts = model.evaluate(torch.tensor(time_series_tensor.reshape(1, -1), dtype=torch.float32).to(device))
            pred, cl, acts = model.evaluate(torch.tensor(
                start.reshape(1, -1), dtype=torch.float32).to(device))
            # pred, cl, acts = model.evaluate(torch.tensor(time_series_tensor.reshape(1, -1), dtype=torch.float32).to(device))
            # pr, cl = torch.topk(torch.tensor(acts,dtype=torch.float32), 2)
            # if verbose == 2:
            #     # print('{}: {:.3f}'.format(0, float(pr[0])))
            #     # print('{}: {:.3f}'.format(1, float(pr[1])))
            #     print("acts",acts)
            # scores[i] = pred[cl]

            scores[i] = acts[0][cl.item()].item()
            print("Scores: ", scores)
            # Render image if verbose, if it's the last step or if save is required.
            if verbose == 2 or (verbose == 1 and i == n_steps) or save_to:
                plt.figure(figsize=(15, 6))
                plt.subplot(121)
                plt.title('{} {:.1f}%, P(1)={:.2f}'.format(
                    ylabel, 100 * i / n_steps, scores[i]))
                plt.plot(list(range(len(X))), X, label="Raw Pattern",
                            color="red", alpha=0.4)
                plt.plot(list(range(len(start))), start,
                            label="Perturbed Pattern")
                plt.scatter(list(sal_order.keys()), list(
                    sal_order.values()), color="orange", label="Saliency")
                plt.xlabel("Timesteps")
                plt.ylabel("Values")
                plt.legend()
                plt.subplot(122)
                plt.plot(np.arange(i+1) / n_steps,
                            scores[:i+1], label="AUC")
                plt.xlim(-0.1, 1.1)
                plt.ylim(0, 1.05)
                plt.fill_between(np.arange(i+1) / n_steps,
                                    0, scores[:i+1], alpha=0.4)
                plt.title(title+"  Minimum Steps  "+str(i))
                plt.xlabel(ylabel)
                plt.ylabel("Accuracy")
                plt.text(x=0.4, y=0.95, s=f'AUC: {auc(scores[:i+1]):.2f}')
                plt.legend()
                if save_to:
                    plt.savefig(save_to + '/{:03d}.png'.format(i))
                    plt.close()
                else:
                    plt.show()
            if i < n_steps:
                coords = salient_order[:,
                                        self.step * i:self.step * (i + 1)][0]
                sal_order[coords[0]] = saliency[int(coords[0])]
                print("sal_order", i, sal_order)
                start.cpu().numpy().reshape(TIMESTEPS)[
                    coords] = finish.cpu().numpy().reshape(TIMESTEPS)[coords]
                # start.cpu().numpy().reshape(TIMESTEPS)[coords] = 0
                # saliency[int(coords[0])] = 0
                # print("Start",start)

            if(self.mode == 'ins' and auc(scores[:i+1]) > self.ins_cutoff):
                print("Best AUC scores in Minimum # of steps for Insertion Metric")
                break
            elif (self.mode == 'del' and auc(scores[:i+1]) < self.del_cutoff):
                print("Best AUC scores in Minimum # of steps for Deletion Metric")
                break
            else:
                pass
        return scores

def qm_plot(X, saliency):
    TIMESTEPS = len(X)
    # noise = torch.tensor(generate_gaussian_noise(X, snrdb=0.001), dtype=torch.float32)
    noise = torch.tensor(np.random.random(size=TIMESTEPS), dtype=torch.float32)
    def blur(x): return x*noise
    model = trained_model
    insertion = CausalMetric(model, 'ins', 1, substrate_fn=blur)
    # insertion = CausalMetric(model, 'ins', 1, substrate_fn=torch.zeros_like)
    #Inverse instead of zeros
    deletion = CausalMetric(model, 'del', 1, substrate_fn=torch.zeros_like)
    scores = insertion.single_run(time_series_tensor=torch.tensor(
        X, dtype=torch.float32), explanation=saliency, verbose=2)
    # scores = insertion.single_run(time_series_tensor=torch.tensor(X, dtype=torch.float32), explanation=saliency, verbose=2)
    # scores = deletion.single_run(time_series_tensor=torch.tensor(X, dtype=torch.float32), explanation=saliency, verbose=2)
    print(scores)
    #Metrics End
