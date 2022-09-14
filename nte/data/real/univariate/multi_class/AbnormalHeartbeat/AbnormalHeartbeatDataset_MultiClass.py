from nte.data.dataset import Dataset
from nte import NTE_MODULE_PATH
import pandas as pd
import os
import numpy as np

# http: // www.peterjbentley.com/heartchallenge/

class AbnormalHeartbeatDataset_MultiClass(Dataset):
    def __init__(self):
        super().__init__(
            name='AbnormalHeartbeat',
            meta_file_path=os.path.join(NTE_MODULE_PATH, 'data', 'real', 'univariate', 'multi_class', 'AbnormalHeartbeat','meta.json'))

    def load_train_data(self):
        df = pd.read_csv(os.path.join(NTE_MODULE_PATH, 'data', 'real', 'univariate', 'multi_class', 'AbnormalHeartbeat', 'train.csv'))
        # df = df.drop(['id'], axis=1)
        train_label = df[df.columns[-1]]
        train_label = train_label.values
        train_label[train_label == 'Normal']     = 0
        train_label[train_label == 'ExtraHLS']   = 1
        train_label[train_label == 'ExtraStole'] = 2
        train_label[train_label == 'Murmur']     = 3
        train_label[train_label == 'Artifact']   = 4
        train_label = train_label.astype(int)
        train_data = df[df.columns[:-1]].to_numpy()
        print("columns are ")
        print(df.columns)
        return train_data, train_label

    def load_test_data(self):
        df = pd.read_csv(os.path.join(NTE_MODULE_PATH, 'data', 'real', 'univariate', 'multi_class', 'AbnormalHeartbeat', 'test.csv'))
        # df = df.drop(['id'], axis=1)
        test_label = df[df.columns[-1]]
        test_label = test_label.values
        test_label[test_label == 'Normal']     = 0
        test_label[test_label == 'ExtraHLS']   = 1
        test_label[test_label == 'ExtraStole'] = 2
        test_label[test_label == 'Murmur']     = 3
        test_label[test_label == 'Artifact']   = 4
        test_label = test_label.astype(int)
        test_data = df[df.columns[:-1]].to_numpy()
        return test_data, test_label
