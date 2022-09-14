# -*- coding: utf-8 -*-
"""
| **@created on:** 11/4/20,
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

class ComputersDataset(Dataset):
    def __init__(self):
        super().__init__(
            name='Computers',
            meta_file_path=os.path.join(NTE_MODULE_PATH, 'data', 'real', 'univariate', 'binary', 'computers', 'meta.json'))

    def load_train_data(self):
        df = pd.read_csv(os.path.join(NTE_MODULE_PATH, 'data', 'real', 'univariate', 'binary', 'computers', 'train.csv'))
        df = df.drop(['id'], axis=1)

        train_data = df[df.columns[:-1]].to_numpy()
        train_label = df[df.columns[-1]]
        train_label = pd.Series([0 if train_label[x] == 1 else 1 for x in range(len(train_label))])
        return train_data, train_label

    def load_test_data(self):
        df = pd.read_csv(os.path.join(NTE_MODULE_PATH, 'data', 'real', 'univariate', 'binary', 'computers', 'test.csv'))
        df = df.drop(['id'], axis=1)

        test_data = df[df.columns[:-1]].to_numpy()
        test_label = df[df.columns[-1]]
        test_label = pd.Series([0 if test_label[x] == 1 else 1 for x in range(len(test_label))])
        return test_data, test_label



# class ComputersDataset(Dataset):
#     def __init__(self):
#         super().__init__()

#         df = pd.read_csv(os.path.join(NTE_MODULE_PATH, 'data', 'real','Computers', 'train.csv'))
#         df.sample(frac=1)
#         df = df.drop(['id'], axis=1)

#         self.train_data = df[df.columns[:-1]].to_numpy()
#         self.train_label = df[df.columns[-1]]
#         self.train_label = pd.Series([0 if self.train_label[x] == 1 else 1 for x in range(len(self.train_label))])
#         print(df.columns)
#         self.class0 = self.train_data[self.train_label == 0]
#         self.class1 = self.train_data[self.train_label == 1]
#         self.class0_mean = self.class0.mean(axis=0)
#         self.class1_mean = self.class1.mean(axis=0)

#         df = pd.read_csv(os.path.join(NTE_MODULE_PATH, 'data', 'real','Computers', 'test.csv'))
#         df = df.drop(['id'], axis=1)
#         df.sample(frac=1)
#         self.test_data = df[df.columns[:-1]].to_numpy()
#         self.test_label = df[df.columns[-1]]
#         self.test_label = pd.Series([0 if self.test_label[x] == -1 else 1 for x in range(len(self.test_label))])
#         del df



