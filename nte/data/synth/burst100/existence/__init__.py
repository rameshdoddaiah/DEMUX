from nte.data.dataset import Dataset
from nte import NTE_MODULE_PATH
import pandas as pd
import os
import random
import numpy as np
from scipy import signal


class BurstExistence(Dataset):
    def __init__(self):
        super().__init__(name='burst_existence')
        try:
            df = pd.read_csv(os.path.join(NTE_MODULE_PATH, 'data', 'burst100', 'existence', 'burst_existence_train.csv'))
            self.train_data = df.drop('label', axis=1).values
            self.train_label = df['label'].values

            df = pd.read_csv(os.path.join(NTE_MODULE_PATH, 'data', 'burst100', 'existence', 'burst_existence_test.csv'))

            self.test_data = df.drop('label', axis=1).values
            self.test_label = df['label'].values
            del df

            self.train_meta, self.test_meta = self.read_meta([os.path.join(NTE_MODULE_PATH, 'data', 'burst100',  'existence','burst_existence_train.meta'),
                                                              os.path.join(NTE_MODULE_PATH, 'data', 'burst100', 'existence', 'burst_existence_test.meta')])

        except Exception as e:
            print(e)
            print(f"Dataset not found {os.path.join(NTE_MODULE_PATH, 'data', 'burst100', 'existence', 'burst_existence_train.csv')}")
            print(f"Dataset not found {os.path.join(NTE_MODULE_PATH, 'data', 'burst100', 'existence', 'burst_existence_test.csv')}")

    def _generate(self, samples):
        """
        """

        def gen(f, samples):
            f.write(','.join(['f' + str(i) for i in range(100)]) + ',' + "label\n")

            t = np.linspace(-1, 1, 100, endpoint=False)
            i, q, e = signal.gausspulse(t, fc=5, retquad=True, retenv=True)
            z = np.zeros([50])
            e = e[25:75]
            candidate_data_1 = lambda : np.insert(z, random.randrange(0, 50), e)
            candidate_data_0 = np.zeros([100])

            for i in range(samples):
                candidate_label = str(random.randrange(0, 2))
                if candidate_label == '0':
                    f.write(','.join(
                        candidate_data_0.astype('str')) + "," + candidate_label + "\n")
                elif candidate_label == '1':
                    f.write(','.join(candidate_data_1().astype('str')) + "," + candidate_label + "\n")

        with open(os.path.join(NTE_MODULE_PATH, 'data', 'burst100', 'existence', 'burst_existence_train.csv'), 'w') as f:
            print(int(samples * 0.8))
            gen(f, int(samples * 0.8))

        with open(os.path.join(NTE_MODULE_PATH, 'data', 'burst100', 'existence', 'burst_existence_test.csv'), 'w') as f:
            print( samples - int(samples * 0.8))
            gen(f, samples - int(samples * 0.8))

        self.create_meta([os.path.join(NTE_MODULE_PATH, 'data', 'burst100', 'existence', 'burst_existence_train.csv'),
                          os.path.join(NTE_MODULE_PATH, 'data', 'burst100', 'existence', 'burst_existence_test.csv')])


if __name__ == '__main__':
    BurstExistence()._generate(1000)
