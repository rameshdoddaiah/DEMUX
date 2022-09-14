from nte.data.dataset import Dataset
from nte import NTE_MODULE_PATH
import pandas as pd
import os
import random
import numpy as np
from scipy import signal


class BurstLocation10(Dataset):
    def __init__(self):
        super().__init__(name='burst_location')
        try:
            df = pd.read_csv(os.path.join(NTE_MODULE_PATH, 'data', 'synth', 'burst10', 'location', 'burst_location_train.csv'))
            self.train_data = df.drop('label', axis=1).values
            self.train_label = df['label'].values

            df = pd.read_csv(os.path.join(NTE_MODULE_PATH, 'data','synth', 'burst10', 'location','burst_location_test.csv'))

            self.test_data = df.drop('label', axis=1).values
            self.test_label = df['label'].values
            del df

            self.train_meta, self.test_meta = self.read_meta([os.path.join(NTE_MODULE_PATH, 'data', 'synth', 'burst10', 'location','burst_location_train.meta'),
                                                              os.path.join(NTE_MODULE_PATH, 'data', 'synth', 'burst10','location', 'burst_location_test.meta')])

        except Exception as e:
            print(e)
            print(f"Dataset not found {os.path.join(NTE_MODULE_PATH, 'data','synth', 'burst10','location', 'burst_location_train.csv')}")
            print(f"Dataset not found {os.path.join(NTE_MODULE_PATH, 'data','synth', 'burst10','location', 'burst_location_test.csv')}")

    def _generate(self, samples):
        """
        """
        def gen(f, samples, train):
            blen = 10
            f.write(','.join(['f' + str(i) for i in range(blen)]) + ',' + "label\n")
            for i in range(samples):
            # for i in range(11):
                candidate_label = str(1)
                # candidate_label = str(random.randrange(0, 2))
                if train and i and (0 == i % 5):
                    candidate_label = str(0)

                if i and (0 == i % 10):
                    candidate_label = str(0)
                # print(i)
                if candidate_label == '0':
                    candidate_data_0 = np.zeros([blen])
                    # candidate_data_0[i%blen] = '1'
                    # print(candidate_data_0)
                    f.write(','.join(
                        candidate_data_0.astype('str')) + "," + candidate_label + "\n")
                elif candidate_label == '1':
                    candidate_data_1 = np.zeros([blen])
                    candidate_data_1[i%blen] = '1'
                    # print(candidate_data_1)
                    f.write(','.join(candidate_data_1.astype('str')) + "," + candidate_label + "\n")

            # for i in range(samples):
            #     candidate_label = str(i%2)
            #     if candidate_label == '0':
            #         candidate_data_0 = np.zeros([blen])
            #         candidate_data_0[i%blen] = '1'
            #         print(candidate_data_0)
            #         f.write(','.join(
            #             candidate_data_0.astype('str')) + "," + candidate_label + "\n")
            #     elif candidate_label == '1':
            #         candidate_data_1 = np.zeros([blen])
            #         candidate_data_1[i%blen] = '1'
            #         # print(candidate_data_1)
            #         f.write(','.join(candidate_data_1.astype('str')) + "," + candidate_label + "\n")
        with open(os.path.join(NTE_MODULE_PATH, 'data', 'burst10','location', 'burst_location_train.csv'), 'w') as f:
            print(int(samples * 0.8))
            gen(f, int(samples * 0.8),1)

        with open(os.path.join(NTE_MODULE_PATH, 'data', 'burst10','location', 'burst_location_test.csv'), 'w') as f:
            print( samples - int(samples * 0.8))
            gen(f, samples - int(samples * 0.8),0)

        self.create_meta([os.path.join(NTE_MODULE_PATH, 'data', 'burst10','location', 'burst_location_train.csv'),
                          os.path.join(NTE_MODULE_PATH, 'data', 'burst10', 'location','burst_location_test.csv')])


if __name__ == '__main__':
    BurstLocation10()._generate(1000)
