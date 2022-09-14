from nte.data.dataset import Dataset
from nte import NTE_MODULE_PATH
import pandas as pd
import os

from nte.data.dataset import Dataset
from nte import NTE_MODULE_PATH
import pandas as pd
import os


class WaferDataset(Dataset):
    def __init__(self):
        super().__init__(
            name='wafer',
            meta_file_path=os.path.join(NTE_MODULE_PATH, 'data', 'real', 'univariate','binary','wafer', 'meta.json'))

    def load_train_data(self):
        df = pd.read_csv(os.path.join(NTE_MODULE_PATH, 'data', 'real', 'univariate','binary','wafer', 'train.csv'), header=None, skiprows=1)

        train_data = df[list(range(152))].values
        train_label = df[152].values
        return train_data, train_label

    def load_test_data(self):
        df = pd.read_csv(os.path.join(NTE_MODULE_PATH, 'data', 'real', 'univariate','binary','wafer', 'test.csv'), header=None, skiprows=1)

        test_data = df[list(range(152))].values
        test_label = df[152].values
        return test_data, test_label



# class WaferDataset(Dataset):
#     def __init__(self, randomize=False):
#         super().__init__()
#         df = pd.read_csv(os.path.join(NTE_MODULE_PATH, 'data', 'real', 'wafer', 'train.csv'), header=None, skiprows=1)
#         if randomize:
#             df.sample(frac=1)
#         self.train_data = df[list(range(152))].values
#         self.train_label = df[152].values
#         self.class0 = self.train_data[self.train_label == 0]
#         self.class1 = self.train_data[self.train_label == 1]
#         self.class0_mean = self.class0.mean(axis=0)
#         self.class1_mean = self.class1.mean(axis=0)
#
#         df = pd.read_csv(os.path.join(NTE_MODULE_PATH, 'data', 'real', 'wafer', 'test.csv'), header=None, skiprows=1)
#         if randomize:
#             df.sample(frac=1)
#         self.test_data = df[list(range(152))].values
#         self.test_label = df[152].values
#         del df
