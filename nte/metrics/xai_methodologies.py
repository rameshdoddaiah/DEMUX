import numpy as np
from nte.data.synth.blipv3.blipv3_dataset import BlipV3Dataset
import pandas as pd
from sklearn.metrics import accuracy_score
import torch

dataset = BlipV3Dataset()

TIMESTEPS = len(dataset.train_data[0])
use_model='dnn'
true_model = torch.load('/Users/prathyushsp/Git/TimeSeriesSaliency/nte/models/blipv3/blip_v3_dnn_mse.ckpt', encoding='latin')

def qm_tcr_random_generate_perturbation_no_subseq_only_input(inp):
    return inp * np.random.randint(2, size=TIMESTEPS)


def evaluate_model(data):
    if not isinstance(data, torch.Tensor):
        data = torch.tensor(data, dtype=torch.float32)
    if use_model=='rnn':
        actual_predictions = true_model(data.reshape([-1, TIMESTEPS, 1]))
    elif use_model=='dnn':
        actual_predictions = true_model(data.reshape([-1, TIMESTEPS]))
    val, predictions = torch.max(actual_predictions, 1)
    return predictions  # val, predictions, actual_predictions

########## RANDOM QM ###########################
# qm(tcr) -> Random Change in Input without use of subsequence
cols = list(range(len(dataset.train_data[0])))
random_tcr_perturbation_values = pd.DataFrame(columns=cols)
for i in range(dataset.test_data.shape[0]):
    pp = pd.Series(qm_tcr_random_generate_perturbation_no_subseq_only_input(dataset.test_data[i]),index=cols)
    random_tcr_perturbation_values = random_tcr_perturbation_values.append(pp,ignore_index=True)

y_pred = evaluate_model(dataset.test_data)

original_test_accuracy = accuracy_score(dataset.test_label, y_pred)
print("QM(T) Original Test data Accuracy No change in input subseq:         %.2f%%" % (original_test_accuracy * 100.0))

y_new_random_tcr_pred = evaluate_model(random_tcr_perturbation_values.values.astype(np.float32))

random_tcr_test_accuracy = accuracy_score(dataset.test_label, y_new_random_tcr_pred)
print("QM(TCR) Random Perturbed data Accuracy No chnage in input subseq:    %.2f%%" % (random_tcr_test_accuracy * 100.0))

