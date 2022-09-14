from nte.data.dataset import Dataset
from nte import NTE_MODULE_PATH
import pandas as pd
import os
import random
import numpy as np
from scipy import signal


class BurstTimeDifferenceStrength10(Dataset):
    def __init__(self):
        super().__init__(name='burst_time_difference_time_difference_strength')
        try:
            df = pd.read_csv(os.path.join(NTE_MODULE_PATH, 'data', 'burst10', 'time_difference_strength', 'burst_time_difference_strength_train.csv'))
            self.train_data = df.drop('label', axis=1).values
            self.train_label = df['label'].values

            df = pd.read_csv(os.path.join(NTE_MODULE_PATH, 'data', 'burst10', 'time_difference_strength','burst_time_difference_strength_test.csv'))

            self.test_data = df.drop('label', axis=1).values
            self.test_label = df['label'].values
            del df

            self.train_meta, self.test_meta = self.read_meta([os.path.join(NTE_MODULE_PATH, 'data', 'burst10', 'time_difference_strength','burst_time_difference_strength_train.meta'),
                                                              os.path.join(NTE_MODULE_PATH, 'data', 'burst10','time_difference_strength', 'burst_time_difference_strength_test.meta')])

        except Exception as e:
            print(e)
            print(f"Dataset not found {os.path.join(NTE_MODULE_PATH, 'data', 'burst10','time_difference_strength', 'burst_time_difference_strength_train.csv')}")
            print(f"Dataset not found {os.path.join(NTE_MODULE_PATH, 'data', 'burst10','time_difference_strength', 'burst_time_difference_strength_test.csv')}")

    def _generate(self, samples):
        """
        """

        def gen(f, samples):
            blen = 10
            f.write(','.join(['f' + str(i) for i in range(blen)]) + ',' + "label\n")

            for i in range(samples):
                candidate_label = str(random.randrange(0, 2))
                if candidate_label == '0':
                    candidate_data_0 = np.zeros([blen])
                    diff0 = 2
                    i = random.randrange(0,blen-diff0)
                    candidate_data_0[i] = '1'
                    candidate_data_0[i+1] = '1'
                    # print(candidate_data_0)
                    # candidate_data_0[(i+diff0)%blen] = '1'
                    f.write(','.join(candidate_data_0.astype('str')) + "," + candidate_label + "\n")
                elif candidate_label == '1':
                    candidate_data_1 = np.zeros([blen])
                    diff1 = 3
                    i = random.randrange(0,blen-diff1)
                    candidate_data_1[i] = '1'
                    candidate_data_1[i+1] = '1'
                    candidate_data_1[i+2] = '1'
                    # candidate_data_1[(i+diff1)%blen-1] = '1'
                    # candidate_data_1[(i+diff1)%blen] = '1'
                    f.write(','.join(candidate_data_1.astype('str')) + "," + candidate_label + "\n")

        with open(os.path.join(NTE_MODULE_PATH, 'data', 'burst10','time_difference_strength', 'burst_time_difference_strength_train.csv'), 'w') as f:
            print(int(samples * 0.8))
            gen(f, int(samples * 0.8))

        with open(os.path.join(NTE_MODULE_PATH, 'data', 'burst10','time_difference_strength', 'burst_time_difference_strength_test.csv'), 'w') as f:
            print( samples - int(samples * 0.8))
            gen(f, samples - int(samples * 0.8))

        self.create_meta([os.path.join(NTE_MODULE_PATH, 'data', 'burst10','time_difference_strength', 'burst_time_difference_strength_train.csv'),
                          os.path.join(NTE_MODULE_PATH, 'data', 'burst10', 'time_difference_strength','burst_time_difference_strength_test.csv')])

if __name__ == '__main__':
    BurstTimeDifferenceStrength10()._generate(1000)