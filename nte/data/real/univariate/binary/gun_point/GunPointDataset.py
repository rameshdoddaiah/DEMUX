from nte.data.dataset import Dataset
from nte import NTE_MODULE_PATH
import pandas as pd
import os


class GunPointDataset(Dataset):
    def __init__(self):
        super().__init__(
            name='gun_point',
            meta_file_path=os.path.join(NTE_MODULE_PATH, 'data', 'real', 'univariate', 'binary', 'gun_point', 'meta.json'))

    def load_train_data(self):
        df = pd.read_csv(os.path.join(NTE_MODULE_PATH, 'data', 'real', 'univariate',
                         'binary', 'gun_point', 'train.csv'), header=None)
        train_data = df[list(range(150))].values
        train_label = df[150].values
        return train_data, train_label

    def load_test_data(self):
        df = pd.read_csv(os.path.join(NTE_MODULE_PATH, 'data', 'real', 'univariate',
                         'binary', 'gun_point', 'test.csv'), header=None)
        test_data = df[list(range(150))].values
        test_label = df[150].values
        return test_data, test_label
