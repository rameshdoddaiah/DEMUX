from tqdm import tqdm
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
import torch
import seaborn as sns
from sklearn.metrics import auc
sns.set_style("darkgrid")


def custom_auc(arr):
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
        pred = torch.softmax(self.model(time_series_tensor), dim=1)

        top, c = torch.max(pred, 0)
        c = c.cpu().numpy()
        n_steps = (TIMESTEPS + self.step - 1) // self.step

        sal_order = {}

        if self.mode == 'del':
            title = 'Deletion Metric'
            ylabel = '% of Pixels deleted'
            start = time_series_tensor.clone()
            finish = self.substrate_fn(time_series_tensor)
        elif self.mode == 'ins':
            title = 'Insertion Metric'
            ylabel = '% of Pixels inserted'
            start = self.substrate_fn(time_series_tensor)
            finish = time_series_tensor.clone()
        else:
            raise Exception('error in mode')

        scores = np.empty(n_steps + 1)
        # Coordinates of pixels in order of decreasing saliency
        salient_order = np.flip(np.argsort(explanation.reshape(-1, TIMESTEPS), axis=1), axis=-1)
        print('Original Prediction: ', model(time_series_tensor), c)
        for i in range(n_steps+1):
            pred = self.model(start)
            pr, cl = torch.topk(pred, 2)
            if verbose == 2:
                print('{}: {:.3f}'.format(0, float(pr[0])))
                print('{}: {:.3f}'.format(1, float(pr[1])))
            scores[i] = pred[c]
            # Render image if verbose, if it's the last step or if save is required.
            if verbose == 2 or (verbose == 1 and i == n_steps) or save_to:
                plt.figure(figsize=(10, 5))
                plt.subplot(121)
                plt.title('{} {:.1f}%, P(1)={:.2f}'.format(ylabel, 100 * i / n_steps, scores[i]))
                plt.plot(list(range(len(start))), start, label="Raw Pattern")
                plt.scatter(list(sal_order.keys()), list(sal_order.values()), color="orange")
                plt.xlabel("Timesteps")
                plt.ylabel("Values")
                plt.legend()
                plt.subplot(122)
                plt.plot(np.arange(i+1) / n_steps, scores[:i+1], label="AUC")
                plt.xlim(-0.1, 1.1)
                plt.ylim(0, 1.05)
                plt.fill_between(np.arange(i+1) / n_steps, 0, scores[:i+1], alpha=0.4)
                plt.title(title)
                plt.xlabel(ylabel)
                plt.ylabel("Accuracy")
                plt.text(x=0.4, y=0.95, s=f'AUC: {custom_auc(scores):.2f}')
                plt.legend()
                if save_to:
                    plt.savefig(save_to + '/{:03d}.png'.format(i))
                    plt.close()
                else:
                    plt.show(i)
            if i < n_steps:
                coords = salient_order[:, self.step * i:self.step * (i + 1)][0]
                # print(coords[0])
                # print(saliency[int(coords[0])])
                sal_order[coords[0]] = explanation[int(coords[0])]
                start.cpu().numpy().reshape(TIMESTEPS)[coords] = finish.cpu().numpy().reshape(TIMESTEPS)[coords]
        return scores

    def evaluate(self, time_series_tensor, exp_batch, batch_size):
        r"""Efficiently evaluate big batch of images.

        Args:
            img_batch (Tensor): batch of images.
            exp_batch (np.ndarray): batch of explanations.
            batch_size (int): number of images for one small batch.

        Returns:
            scores (nd.array): Array containing scores at every step for every image.
        """
        n_samples = time_series_tensor.shape[0]
        n_classes = 2
        TIMESTEPS = time_series_tensor.shape[1]
        predictions = torch.FloatTensor(n_samples, n_classes)
        assert n_samples % batch_size == 0
        for i in tqdm(range(n_samples // batch_size), desc='Predicting labels'):
            preds = torch.softmax(self.model(time_series_tensor[i*batch_size:(i+1)*batch_size]).cpu(), dim=1)
            predictions[i*batch_size:(i+1)*batch_size] = preds
        top = np.argmax(predictions.detach().numpy(), -1)
        n_steps = (TIMESTEPS + self.step - 1) // self.step
        scores = np.zeros((n_steps + 1, n_samples))
        salient_order = np.flip(np.argsort(exp_batch.reshape(-1, TIMESTEPS), axis=1), axis=-1)
        r = np.arange(n_samples).reshape(n_samples, 1)

        substrate = torch.zeros_like(time_series_tensor)
        for j in tqdm(range(n_samples // batch_size), desc='Substrate'):
            substrate[j*batch_size:(j+1)*batch_size] = self.substrate_fn(time_series_tensor[j*batch_size:(j+1)*batch_size])

        if self.mode == 'del':
            caption = 'Deleting  '
            start = time_series_tensor.clone()
            finish = substrate
        elif self.mode == 'ins':
            caption = 'Inserting '
            start = substrate
            finish = time_series_tensor.clone()

        # While not all pixels are changed
        for i in tqdm(range(n_steps+1), desc=caption + 'timesteps'):
            # Iterate over batches
            for j in range(n_samples // batch_size):
                # Compute new scores
                preds = torch.softmax(self.model(start[j*batch_size:(j+1)*batch_size]), dim=1)
                preds = preds.detach().numpy()[range(batch_size), top[j*batch_size:(j+1)*batch_size]]
                scores[i, j*batch_size:(j+1)*batch_size] = preds
            # Change specified number of most salient pixels to substrate pixels
            coords = salient_order[:, self.step * i:self.step * (i + 1)]
            start.cpu().numpy().reshape(n_samples, TIMESTEPS)[r, coords] = finish.cpu().numpy().reshape(n_samples, TIMESTEPS)[r, coords]
        scores = scores.mean(1)
        print(f'Mode:{self.mode.upper()} | TrapAUC:{auc(x=list(range(len(scores))),y=scores)/((len(scores)-1)):.2f} | DiagAUC: {custom_auc(scores):.2f}')
        return scores
if __name__ == '__main__':
    # X = np.array([0,1,1,1,0,0,1,1,0,0])
    # X_BATCH = np.array([[0,1,1,1,0,0,1,1,0,0],[0,1,1,1,0,0,1,1,0,0]])
    # saliency = np.array([[0,0.7,0.9,0.6,0,0,0,0,0,0], [0,0.7,0.9,0.6,0,0,0,0,0,0]])
    # noise = torch.tensor(generate_gaussian_noise(X, snrdb=0.001), dtype=torch.float32)

    from nte.saliencies.blipv3 import *
    from nte.saliencies.wafer import *
    # from nte.models import WaferDNNModel
    from nte.data.synth.burst100 import BurstExistence
    from nte.saliencies.burst100 import *

    # data = BlipV3Dataset()
    # data = WaferDataset()
    data = BurstExistence()
    X_BATCH = data.test_data
    # saliency = Wafer_DNN_MSE_SHAP_Saliency()
    saliency = BurstExistence_DNN_CE_SHAP_Saliency()
    TIMESTEPS = X_BATCH.shape[1]
    n_classes = 2

    noise = torch.tensor(np.random.normal(size=TIMESTEPS), dtype=torch.float32)
    blur = lambda x: x + noise

    # model = BlipV3DNNModel()
    # model = WaferDNNModel()

    from nte.trained_models.burst100 import BurstExistenceDNNModel

    model = BurstExistenceDNNModel()

    # def model(inp):
    #     inp=inp.detach().numpy().flatten()
    #     if inp[1]==1 and inp[2]==1 and inp[3]==1:
    #         return torch.tensor([0.01,0.99])
    #     else:
    #         return torch.tensor([0,0.01])
    insertion = CausalMetric(model, 'ins', 1, substrate_fn=blur)
    deletion = CausalMetric(model, 'del', 1, substrate_fn=torch.zeros_like)

    # scores = insertion.single_run(time_series_tensor=torch.tensor(X_BATCH[0], dtype=torch.float32), explanation=saliency[0], verbose=2)
    # scores = deletion.single_run(time_series_tensor=torch.tensor(X, dtype=torch.float32), explanation=saliency, verbose=2)
    insertion.evaluate(time_series_tensor=torch.tensor(X_BATCH, dtype=torch.float32), exp_batch=saliency, batch_size=8)
    deletion.evaluate(time_series_tensor=torch.tensor(X_BATCH, dtype=torch.float32), exp_batch=saliency, batch_size=8)
    # print(scores)

    import seaborn as sns

    sns.set_style()