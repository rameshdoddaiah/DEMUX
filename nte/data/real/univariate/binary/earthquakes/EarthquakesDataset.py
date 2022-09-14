from nte.data.dataset import Dataset
from nte import NTE_MODULE_PATH
import pandas as pd
import os


class EarthquakesDataset(Dataset):
    def __init__(self):
        super().__init__(
            name='Earthquake',
            meta_file_path=os.path.join(NTE_MODULE_PATH, 'data', 'real', 'earthquakes', 'meta.json'))

    def load_train_data(self):
        df = pd.read_csv(os.path.join(NTE_MODULE_PATH, 'data', 'real', 'earthquakes', 'train.csv'))
        df = df.drop(['id'], axis=1)

        train_data = df[df.columns[:-1]].to_numpy()
        train_label = df[df.columns[-1]]
        train_label = pd.Series([0 if train_label[x] == 0 else 1 for x in range(len(train_label))])
        return train_data, train_label

    def load_test_data(self):
        df = pd.read_csv(os.path.join(NTE_MODULE_PATH, 'data', 'real', 'earthquakes', 'test.csv'))
        df = df.drop(['id'], axis=1)

        test_data = df[df.columns[:-1]].to_numpy()
        test_label = df[df.columns[-1]]
        test_label = pd.Series([0 if test_label[x] == 0 else 1 for x in range(len(test_label))])
        return test_data, test_label
