from nte.data.dataset import Dataset
from nte import NTE_MODULE_PATH
import pandas as pd
import os
import numpy as np

# http: // www.peterjbentley.com/heartchallenge/

class ACSFDataset_MultiClass(Dataset):
    def __init__(self):
        super().__init__(
            name='ACSF',
            meta_file_path=os.path.join(NTE_MODULE_PATH, 'data', 'real', 'univariate', 'multi_class', 'ACSF1','meta.json'))

    def load_train_data(self):
        df = pd.read_csv(os.path.join(NTE_MODULE_PATH, 'data', 'real', 'univariate', 'multi_class', 'ACSF1', 'train.csv'))
        # df = df.drop(['id'], axis=1)
        train_label = df[df.columns[-1]]
        train_label = train_label.values
        # train_label = train_label-1
        train_data = df[df.columns[:-1]].to_numpy()
        print("columns are ")
        print(df.columns)
        return train_data, train_label

    def load_test_data(self):
        df = pd.read_csv(os.path.join(NTE_MODULE_PATH, 'data', 'real', 'univariate', 'multi_class', 'ACSF1', 'test.csv'))
        # df = df.drop(['id'], axis=1)
        test_label = df[df.columns[-1]]
        test_label = test_label.values
        # test_label = test_label-1
        test_data = df[df.columns[:-1]].to_numpy()
        return test_data, test_label
