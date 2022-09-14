# -*- coding: utf-8 -*-
"""
| **@created on:** 3/3/21,
| **@author:** prathyushsp,
| **@version:** v0.0.1
|
| **Description:**
| 
|
| **Sphinx Documentation Status:** 
"""


from nte.data.dataset import Dataset
from nte import NTE_MODULE_PATH
import pandas as pd
import os
from scipy.io import arff
import pandas as pd



class EEGDataset(Dataset):
    def __init__(self):
        super().__init__(
            name='eeg',
            meta_file_path=os.path.join(NTE_MODULE_PATH, 'data', 'real', 'wafer', 'meta.json'))

    def load_train_data(self):
        df = pd.read_csv(os.path.join(NTE_MODULE_PATH, 'data', 'real', 'wafer', 'train.csv'), header=None, skiprows=1)

        train_data = df[list(range(152))].values
        train_label = df[152].values
        return train_data, train_label

    def load_test_data(self):
        df = pd.read_csv(os.path.join(NTE_MODULE_PATH, 'data', 'real', 'wafer', 'test.csv'), header=None, skiprows=1)

        test_data = df[list(range(152))].values
        test_label = df[152].values
        return test_data, test_label