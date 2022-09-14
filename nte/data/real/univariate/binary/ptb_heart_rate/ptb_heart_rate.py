from nte.data.dataset import Dataset
from nte import NTE_MODULE_PATH
import pandas as pd
import os


class PTBHeartRateDataset(Dataset):
    def __init__(self):
        super().__init__(
            name='PTBHeartRate',
            meta_file_path=os.path.join(NTE_MODULE_PATH, 'data', 'real', 'ptb_heart_rate', 'meta.json'))

    def load_train_data(self):
        df = pd.read_csv(os.path.join(NTE_MODULE_PATH, 'data', 'real','ptb_heart_rate', 'train.csv'), header=None, index_col=0)
        train_data = df[list(range(1,188))].values
        train_label = df[189].values
        return train_data, train_label

    def load_test_data(self):
        # df = pd.read_csv(os.path.join(NTE_MODULE_PATH, 'data', 'ptb_heart_rate', 'test.csv'), header=None, index_col=0)
        df = pd.read_csv(os.path.join(NTE_MODULE_PATH, 'data', 'real', 'ptb_heart_rate', 'train.csv'), header=None, index_col=0)
        test_data = df[list(range(1,188))].values
        test_label = df[189].values
        return test_data, test_label
