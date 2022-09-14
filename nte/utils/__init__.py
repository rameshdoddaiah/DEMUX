import numpy as np
import json
import hashlib
import torch
from sklearn.metrics import accuracy_score
from collections import defaultdict

# class RainbowPrint():
#     dark_violet = fg(5, 152, 154)
#     dark_indigo = fg(179, 128, 168)
#     dark_blue = fg(112, 154, 180)
#     dark_green = fg(116, 162, 103)
#     dark_yellow = fg(255, 255, 0)
#     dark_orange = fg(255, 127, 0)
#     dark_red = fg(255, 0, 0)

#     dark_rainbow_colors = [dark_violet, dark_green, dark_indigo, dark_orange, dark_blue, dark_yellow, dark_red]

#     light_violet = fg(54, 54, 54)
#     light_indigo = fg(139, 123, 139)
#     light_blue = fg(111, 153, 180)
#     light_green = fg(85, 107, 46)
#     light_yellow = fg(128, 165, 32)
#     light_orange = fg(255, 127, 0)
#     light_red = fg(173, 71, 71)

#     light_rainbow_colors = [light_violet, light_blue, light_green, light_indigo, light_orange, light_yellow, light_red]

#     def __init__(self, theme="light"):
#         print('theme:', theme)
#         if theme == 'light':
#             self.rainbow_colors = self.light_rainbow_colors
#         else:
#             self.rainbow_colors = self.dark_rainbow_colors

#     def __call__(self, string: str = None, data_dict: dict = None, sep: str = '|'):
#         if string is not None:
#             pass
#         elif data_dict is not None:
#             str_builder = ''
#             for e, (k, v) in enumerate(data_dict.items()):
#                 str_builder += self.rainbow_colors[int(e % 7)] + f"{k}: {str(v)}" + fg.rs + ' | '
#             print(str_builder)
#         else:
#             raise Exception('Either provide string or data_dict. No data to print')


# printr = RainbowPrint()


def rounder(arr):
    t = []
    for a in arr:
        t.append(round(a, 2))
    return t

def normalize(saliency):
    return (abs(saliency) + 1e-5) / (max(abs(saliency)) + 1e-5)

# def confidence_score_v1(predictions, labels)

def confidence_score(predictions, labels):
    correct_indices= defaultdict(lambda : []) 

    for e,(p, l) in enumerate(zip(predictions, labels)):
        if np.argmax(p) == l:
            correct_indices[l].append(e)

    scores = {}
    for k,v in correct_indices.items():
        if len(v)>0:
            scores[k]=np.max(predictions[v],1).mean().flatten()
    return scores, np.mean(list(scores.values()))


def accuracy(true_val, pred_val, threshold=torch.FloatTensor([0.5])):
    out = (pred_val > threshold).float() * 1
    return accuracy_score(true_val, out)


def accuracy_softmax(true_val, pred_val):
    return accuracy_score(torch.max(pred_val, 1)[1].cpu().detach().numpy(), true_val.cpu().detach().numpy())


def accuracy_softmax_mse(true_val, pred_val):
    return accuracy_score(torch.max(pred_val, 1)[1].cpu().detach().numpy(), torch.max(true_val, 1)[1].cpu().detach().numpy())


def get_md5_checksum(file_list):
    md5sum = []
    for file in file_list:
        hasher = hashlib.md5()
        with open(file, 'rb') as afile:
            buf = afile.read()
            hasher.update(buf)
        md5sum.append(hasher.hexdigest())
    return md5sum


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
    noise_volts = np.random.normal(mean_noise, np.sqrt(noise_avg_watts), size=x_watts.shape)
    # Noise up the original signal
    return noise_volts


def softmax(X, theta=1.0, axis=None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: ND-Array. Probably should be floats.
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter,
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis=axis), axis)

    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis=axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p


def find_unique_candidates(data, labels):
    zi = np.argwhere(labels == 0).flatten()
    oi = np.argwhere(labels == 1).flatten()
    euclid_dist = {}
    for zc in zi:
        for oc in oi:
            euclid_dist[f"{zc},{oc}"] = np.linalg.norm(data[zc] - data[oc])
    t1 = [int(i) for i in max(euclid_dist, key=lambda key: euclid_dist[key]).split(',')]
    t2 = [int(i) for i in min(euclid_dist, key=lambda key: euclid_dist[key]).split(',')]

    t3 = [np.mean(np.take(data, np.argwhere(labels == 0).flatten(), axis=0), axis=0),
          np.mean(np.take(data, np.argwhere(labels == 1).flatten(), axis=0), axis=0)]
    return {"max": t1, "min": t2, "avg": t3}


class CustomJsonEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """

    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, set):
            return list(obj)
        else:
            try:
                return obj.default()
            except Exception:
                return f'Object not serializable - {obj}'



import traceback
import sys

# Context manager that copies stdout and any exceptions to a log file
class Tee(object):
    def __init__(self, filename):
        self.file = open(filename, 'w')
        self.stdout = sys.stdout

    def __enter__(self):
        sys.stdout = self

    def __exit__(self, exc_type, exc_value, tb):
        sys.stdout = self.stdout
        if exc_type is not None:
            self.file.write(traceback.format_exc())
        self.file.close()

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()
