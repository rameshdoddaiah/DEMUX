from nte.data.dataset import Dataset
from nte import NTE_MODULE_PATH
import pandas as pd
import os
import numpy as np


class CricketXDataset(Dataset):
    def __init__(self):
        super().__init__(
            name='CricketX',
            meta_file_path=os.path.join(NTE_MODULE_PATH, 'data','real','univariate', 'binary', 'cricketx', 'meta.json'))

    def load_train_data(self):
        df = pd.read_csv(os.path.join(NTE_MODULE_PATH, 'data', 'real','univariate', 'binary','cricketx', 'train.csv'))
        df = df.drop(['id'], axis=1)
        train_data = df[df.columns[:-1]].to_numpy()
        train_label = df[df.columns[-1]]
        train_label = pd.Series([0 if train_label[x] == 1 else 1 if train_label[x] == 2 else 2 for x in range(len(train_label))])
        train_label = pd.Series([0 if train_label[x] == 1 else 1 if train_label[x] == 2 else 2 for x in range(len(train_label))])
        idx = np.where(train_label < 2)[0]
        train_label = train_label.values[idx]
        train_data = df[df.columns[:-1]].to_numpy()[idx]
        return train_data, train_label

    def load_test_data(self):
        df = pd.read_csv(os.path.join(NTE_MODULE_PATH, 'data', 'real','univariate', 'binary','cricketx', 'test.csv'))
        df = df.drop(['id'], axis=1)
        test_data = df[df.columns[:-1]].to_numpy()
        test_label = df[df.columns[-1]]
        test_label = pd.Series([0 if test_label[x] == 1 else 1 if test_label[x] == 2 else 2 for x in range(len(test_label))])
        idx = np.where(test_label != 2)[0]
        test_label = test_label.values[idx]
        test_data = df[df.columns[:-1]].to_numpy()[idx]
        return test_data, test_label



