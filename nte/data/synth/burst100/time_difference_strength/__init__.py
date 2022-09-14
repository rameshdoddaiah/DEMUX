from nte.data.dataset import Dataset
from nte import NTE_MODULE_PATH
import pandas as pd
import os
import random
import numpy as np
from scipy import signal


class BurstTimeDifferenceStrength(Dataset):
    def __init__(self):
        super().__init__(name='burst_time_difference_time_difference_strength')
        try:
            df = pd.read_csv(os.path.join(NTE_MODULE_PATH, 'data', 'burst100', 'time_difference_strength', 'burst_time_difference_strength_train.csv'))
            self.train_data = df.drop('label', axis=1).values
            self.train_label = df['label'].values

            df = pd.read_csv(os.path.join(NTE_MODULE_PATH, 'data', 'burst100', 'time_difference_strength','burst_time_difference_strength_test.csv'))

            self.test_data = df.drop('label', axis=1).values
            self.test_label = df['label'].values
            del df

            self.train_meta, self.test_meta = self.read_meta([os.path.join(NTE_MODULE_PATH, 'data', 'burst100', 'time_difference_strength','burst_time_difference_strength_train.meta'),
                                                              os.path.join(NTE_MODULE_PATH, 'data', 'burst100','time_difference_strength', 'burst_time_difference_strength_test.meta')])

        except Exception as e:
            print(e)
            print(f"Dataset not found {os.path.join(NTE_MODULE_PATH, 'data', 'burst100','time_difference_strength', 'burst_time_difference_strength_train.csv')}")
            print(f"Dataset not found {os.path.join(NTE_MODULE_PATH, 'data', 'burst100','time_difference_strength', 'burst_time_difference_strength_test.csv')}")

    def _generate(self, samples):
        """
        """

        def gen(f, samples):
            f.write(','.join(['f' + str(i) for i in range(100)]) + ',' + "label\n")

            t = np.linspace(-1, 1, 100, endpoint=False)
            _, _, e1 = signal.gausspulse(t, fc=10, retquad=True, retenv=True)
            _, _, e2 = signal.gausspulse(t, fc=10, retquad=True, retenv=True)
            z = np.zeros([40])
            e1 = e1[35:65]
            e2 = e2[35:65] / 2
            c1 = np.concatenate((e1, e2))
            c2 = np.concatenate((e2, e1))
            candidate_data_1 = lambda : np.insert(z, random.randrange(0, 40), c1)
            candidate_data_0 = lambda : np.insert(z, random.randrange(0, 40), c2)

            for i in range(samples):
                candidate_label = str(random.randrange(0, 2))
                if candidate_label == '0':
                    f.write(','.join(candidate_data_0().astype('str')) + "," + candidate_label + "\n")
                elif candidate_label == '1':
                    f.write(','.join(candidate_data_1().astype('str')) + "," + candidate_label + "\n")

        with open(os.path.join(NTE_MODULE_PATH, 'data', 'burst100','time_difference_strength', 'burst_time_difference_strength_train.csv'), 'w') as f:
            print(int(samples * 0.8))
            gen(f, int(samples * 0.8))

        with open(os.path.join(NTE_MODULE_PATH, 'data', 'burst100','time_difference_strength', 'burst_time_difference_strength_test.csv'), 'w') as f:
            print( samples - int(samples * 0.8))
            gen(f, samples - int(samples * 0.8))

        self.create_meta([os.path.join(NTE_MODULE_PATH, 'data', 'burst100','time_difference_strength', 'burst_time_difference_strength_train.csv'),
                          os.path.join(NTE_MODULE_PATH, 'data', 'burst100', 'time_difference_strength','burst_time_difference_strength_test.csv')])

if __name__ == '__main__':
    BurstTimeDifferenceStrength._generate(1000)