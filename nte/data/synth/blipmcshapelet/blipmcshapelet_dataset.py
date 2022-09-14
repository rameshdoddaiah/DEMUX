from nte.data.dataset import Dataset
from nte import NTE_MODULE_PATH
import pandas as pd
import os
import random
import numpy as np


class BlipMCShapeletDataset(Dataset):
    def __init__(self):
        super().__init__(name='blipmcshapelet',
                         meta_file_path=os.path.join(NTE_MODULE_PATH, 'data', 'synth', 'blipmcshapelet', 'meta.json'))

    def load_train_data(self):
        df = pd.read_csv(os.path.join(NTE_MODULE_PATH, 'data', 'synth', 'blipmcshapelet', 'train.csv'), header=0)

        train_data = df.drop('label', axis=1).values
        train_label = df['label'].values
        return train_data, train_label

    def load_test_data(self):
        df = pd.read_csv(os.path.join(NTE_MODULE_PATH, 'data', 'synth', 'blipmcshapelet', 'test.csv'), header=0)

        test_data = df.drop('label', axis=1).values
        test_label = df['label'].values

        # uniques = np.unique(test_data, return_index=True, axis=0)
        # test_data = test_data[uniques[1]]
        # test_label = test_label[uniques[1]]
        return test_data, test_label

    def _generate(self, samples):
        """
        10 Timesteps
        0 1 1 1 0 0 1 1 0 0 - 1
        0 0 0 0 0 0 1 1 0 0 - 0
        """

        def gen(f, samples, test=False):
            f.write(','.join(['f' + str(i) for i in range(10)]) + ',' + "label\n")

            # candidate_data_0 = [['1', '1', '0', '0', '0', '0', '0', '0', '0', '0'],
            #                     ['0', '1', '1', '0', '0', '0', '0', '0', '0', '0'],
            #                     ['0', '0', '1', '1', '0', '0', '0', '0', '0', '0'],
            #                     ['0', '0', '0', '1', '1', '0', '0', '0', '0', '0'],
            #                     ['0', '0', '0', '0', '1', '1', '0', '0', '0', '0'],
            #                     ['0', '0', '0', '0', '0', '1', '1', '0', '0', '0'],
            #                     ['0', '0', '0', '0', '0', '0', '1', '1', '0', '0'],
            #                     ['0', '0', '0', '0', '0', '0', '0', '1', '1', '0'],
            #                     ['0', '0', '0', '0', '0', '0', '0', '0', '1', '1']]

            candidate_data_0 = [['1', '1', '1', '0', '0', '0', '0', '0', '0', '0'],
                                ['0', '1', '1', '1', '0', '0', '0', '0', '0', '0'],
                                ['0', '0', '1', '1', '1', '0', '0', '0', '0', '0'],
                                ['0', '0', '0', '1', '1', '1', '0', '0', '0', '0'],
                                ['0', '0', '0', '0', '1', '1', '1', '0', '0', '0'],
                                ['0', '0', '0', '0', '0', '1', '1', '1', '0', '0'],
                                ['0', '0', '0', '0', '0', '0', '1', '1', '1', '0'],
                                ['0', '0', '0', '0', '0', '0', '0', '1', '1', '1']]


            candidate_data_1 = [['1', '0', '1', '0', '0', '0', '0', '0', '0', '0'],
                                ['0', '1', '0', '1', '0', '0', '0', '0', '0', '0'],
                                ['0', '0', '1', '0', '1', '0', '0', '0', '0', '0'],
                                ['0', '0', '0', '1', '0', '1', '0', '0', '0', '0'],
                                ['0', '0', '0', '0', '1', '0', '1', '0', '0', '0'],
                                ['0', '0', '0', '0', '0', '1', '0', '1', '0', '0'],
                                ['0', '0', '0', '0', '0', '0', '1', '0', '1', '0'],
                                ['0', '0', '0', '0', '0', '0', '0', '1', '0', '1']]

            if test == True:
                for i in range(samples):
                    candidate_label = str(random.randrange(0, 2))
                    if candidate_label == '0':
                        f.write(','.join(candidate_data_0[random.randint(0, len(candidate_data_0)) - 1]) + "," + candidate_label + "\n")
                    elif candidate_label == '1':
                        f.write(','.join(candidate_data_1[random.randint(0, len(candidate_data_1)) - 1]) + "," + candidate_label + "\n")
            else:
                for i in range(samples):
                    candidate_label = str(random.randrange(0, 2))
                    if candidate_label == '0':
                        f.write(','.join(candidate_data_0[random.randint(0, len(candidate_data_0)) - 1]) + "," + candidate_label + "\n")
                    elif candidate_label == '1':
                        f.write(','.join(candidate_data_1[random.randint(0, len(candidate_data_1)) - 1]) + "," + candidate_label + "\n")

        with open(os.path.join(NTE_MODULE_PATH, 'data','synth', 'blipmc', 'train.csv'), 'w') as f:
            print(int(samples * 0.8))
            gen(f, int(samples * 0.8),test=False)

        with open(os.path.join(NTE_MODULE_PATH, 'data', 'synth','blipmc', 'test.csv'), 'w') as f:
            print(samples - int(samples * 0.8))
            gen(f, samples - int(samples * 0.8),test=True)

        self.create_meta([os.path.join(NTE_MODULE_PATH, 'data', 'synth','blipmc', 'train.csv'), os.path.join(NTE_MODULE_PATH, 'data', 'synth','blipmc', 'test.csv')])


if __name__ == '__main__':
    BlipMCShapeletDataset()._generate(1000)
