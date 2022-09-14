from nte.data.dataset import Dataset
from nte import NTE_MODULE_PATH
import pandas as pd
import os
import random
import numpy as np
from scipy import signal


class BurstFrequency10(Dataset):
    def __init__(self):
        super().__init__(name='burst_frequency')
        try:
            df = pd.read_csv(os.path.join(NTE_MODULE_PATH, 'data', 'burst10', 'frequency', 'burst_frequency_train.csv'))
            self.train_data = df.drop('label', axis=1).values
            self.train_label = df['label'].values

            df = pd.read_csv(os.path.join(NTE_MODULE_PATH, 'data', 'burst10', 'frequency','burst_frequency_test.csv'))

            self.test_data = df.drop('label', axis=1).values
            self.test_label = df['label'].values
            del df

            self.train_meta, self.test_meta = self.read_meta([os.path.join(NTE_MODULE_PATH, 'data', 'burst10', 'frequency','burst_frequency_train.meta'),
                                                              os.path.join(NTE_MODULE_PATH, 'data', 'burst10','frequency', 'burst_frequency_test.meta')])

        except Exception as e:
            print(e)
            print(f"Dataset not found {os.path.join(NTE_MODULE_PATH, 'data', 'burst10','frequency', 'burst_frequency_train.csv')}")
            print(f"Dataset not found {os.path.join(NTE_MODULE_PATH, 'data', 'burst10','frequency', 'burst_frequency_test.csv')}")

    def _generate(self, samples):
        """
        """

        def gen(f, samples):
            blen = 10
            f.write(','.join(['f' + str(i) for i in range(blen)]) + ',' + "label\n")

            def gen_candidate_1():
                z1 = np.zeros([blen])
                i = random.randrange(0,blen)
                j = random.randrange(0,blen)
                z1[i] = '1'
                if i != j:
                    z1[j] = '1'
                else:
                    z1[(i+1)%blen] = '1'
                return z1

            def gen_candidate_0():
                z2 = np.zeros([blen])
                i = random.randrange(0,blen)
                z2[i] = '1'
                return z2

            candidate_data_1 = lambda : gen_candidate_1()
            candidate_data_0 = lambda : gen_candidate_0()

            for i in range(samples):
                candidate_label = str(random.randrange(0, 2))
                if candidate_label == '0':
                    f.write(','.join(candidate_data_0().astype('str')) + "," + candidate_label + "\n")
                elif candidate_label == '1':
                    print(candidate_data_1())
                    f.write(','.join(candidate_data_1().astype('str')) + "," + candidate_label + "\n")

        with open(os.path.join(NTE_MODULE_PATH, 'data', 'burst10','frequency', 'burst_frequency_train.csv'), 'w') as f:
            print(int(samples * 0.8))
            gen(f, int(samples * 0.8))

        with open(os.path.join(NTE_MODULE_PATH, 'data', 'burst10','frequency', 'burst_frequency_test.csv'), 'w') as f:
            print( samples - int(samples * 0.8))
            gen(f, samples - int(samples * 0.8))

        self.create_meta([os.path.join(NTE_MODULE_PATH, 'data', 'burst10','frequency', 'burst_frequency_train.csv'),
                          os.path.join(NTE_MODULE_PATH, 'data', 'burst10', 'frequency','burst_frequency_test.csv')])


if __name__ == '__main__':
    BurstFrequency10()._generate(1000)
