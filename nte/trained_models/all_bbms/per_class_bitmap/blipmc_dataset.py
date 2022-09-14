from nte.data.dataset import Dataset
from nte import NTE_MODULE_PATH
import pandas as pd
import os
import random
import numpy as np


class BlipMCDataset(Dataset):
    def __init__(self):
        super().__init__(name='blip',
                         meta_file_path=os.path.join(NTE_MODULE_PATH, 'data', 'synth', 'blipmc', 'meta.json'))

    def load_train_data(self):
        df = pd.read_csv(os.path.join(NTE_MODULE_PATH, 'data', 'synth', 'blipmc', 'train.csv'), header=0)

        train_data = df.drop('label', axis=1).values
        train_label = df['label'].values
        return train_data, train_label

    def load_test_data(self):
        df = pd.read_csv(os.path.join(NTE_MODULE_PATH, 'data', 'synth', 'blipmc', 'test.csv'), header=0)

        test_data = df.drop('label', axis=1).values
        test_label = df['label'].values

        uniques = np.unique(test_data, return_index=True, axis=0)
        test_data = test_data[uniques[1]]
        test_label = test_label[uniques[1]]
        return test_data, test_label

    def _generate(self, samples):
        """
        10 Timesteps
        0 1 1 1 0 0 1 1 0 0 - 1
        0 0 0 0 0 0 1 1 0 0 - 0
        """

        def gen(f, samples):
            f.write(','.join(['f' + str(i) for i in range(10)]) + ',' + "label\n")

            # candidate_data_10 = [
            #     ['0', '1', '1', '1', '0', '0', '1', '1', '0', '0']]
            # candidate_data_11 = [
            #     ['1', '0', '1', '0', '1', '0', '1', '1', '0', '0']]

            # candidate_data_12 = [
            #     ['1', '1', '0', '0', '1', '1', '1', '1', '0', '0']]

            # candidate_data_00 = [
            #     ['0', '0', '0', '0', '0', '0', '1', '1', '0', '0']]

            # candidate_data_01 = [
            #     ['1', '1', '1', '1', '1', '1', '1', '1', '0', '0']]

            # candidate_data_02 = [
            #     ['0', '1', '0', '1', '0', '1', '1', '1', '0', '0']]

            # candidate_data_20 = [
            #     ['0', '0', '0', '0', '0', '0', '1', '1', '0', '0']]
            # candidate_data_21 = [
            #     ['0', '0', '0', '0', '0', '0', '1', '1', '0', '0']]

            # candidate_data_22 = [
            #     ['0', '0', '0', '0', '0', '0', '1', '1', '0', '0']]

            # candidate_data_0 = [['100', '2', '2', '2', '2', '2', '2', '2', '2', '2']]
            # candidate_data_1 = [['2', '100', '2', '2', '2', '2', '2', '2', '2', '2']]
            # candidate_data_2 = [['2', '2', '100', '2', '2', '2', '2', '2', '2', '2']]
            # candidate_data_3 = [['2', '2', '2', '100', '2', '2', '2', '2', '2', '2']]
            # candidate_data_4 = [['2', '2', '2', '2', '100', '2', '2', '2', '2', '2']]
            candidate_data_0 = [['1', '0', '0', '0', '0', '0', '0', '0', '0', '0']]
            candidate_data_1 = [['0', '1', '0', '0', '0', '0', '0', '0', '0', '0']]
            candidate_data_2 = [['0', '0', '1', '0', '0', '0', '0', '0', '0', '0']]
            candidate_data_3 = [['0', '0', '0', '1', '0', '0', '0', '0', '0', '0']]
            candidate_data_4 = [['0', '0', '0', '0', '1', '0', '0', '0', '0', '0']]
            candidate_data_5 = [['0', '0', '0', '0', '0', '1', '0', '0', '0', '0']]
            candidate_data_6 = [['0', '0', '0', '0', '0', '0', '1', '0', '0', '0']]
            candidate_data_7 = [['0', '0', '0', '0', '0', '0', '0', '1', '0', '0']]
            candidate_data_8 = [['0', '0', '0', '0', '0', '0', '0', '0', '1', '0']]
            candidate_data_9 = [['0', '0', '0', '0', '0', '0', '0', '0', '0', '1']]

            # candidate_data_0 = [['0.1', '1', '0.1', '0.1', '0.1', '0.1', '0.1', '0.1', '0.1', '0.1']]
            # candidate_data_1 = [['0.1', '1', '0.1', '0.1', '0.1', '0.1', '0.1', '0.1', '0.1', '0.1']]
            # candidate_data_2 = [['0.1', '0.1', '1', '0.1', '0.1', '0.1', '0.1', '0.1', '0.1', '0.1']]
            # candidate_data_3 = [['0.1', '0.1', '0.1', '1', '0.1', '0.1', '0.1', '0.1', '0.1', '0.1']]
            # candidate_data_4 = [['0.1', '0.1', '0.1', '0.1', '1', '0.1', '0.1', '0.1', '0.1', '0.1']]
            # candidate_data_4 = [['0', '0', '0', '0', '0', '0', '1', '1', '1', '1']]

            # candidate_data_3 = [['1', '0', '0', '0', '1', '0', '1', '1', '1', '1']]
            # ###################   0    1    2    3    4    5    6    7    8    9
            # #frog 0.05
            # candidate_data_2 = [['1', '1', '0', '0', '0', '0', '1', '1', '1', '1']]
            # ###################   0    1    2    3    4    5    6    7    8    9

            # #dog #0.3 given probability  desired probability 0.9 or close to 1.0
            # candidate_data_1 = [['0', '1', '1', '1', '0', '0', '1', '1', '0', '0']]
            # ###################   0    1    2    3    4    5    6    7    8    9

            # #cat 0.6 probability
            # #Mask bitmap, 10 time steps . bitmap 10, 
            # candidate_data_0 = [['0', '1', '0', '0', '0', '0', '1', '1', '0', '0'],
            #                     ['0', '0', '1', '0', '0', '0', '1', '1', '0', '0'],
            #                     ['0', '0', '0', '1', '0', '0', '1', '1', '0', '0'],
            #                     ['0', '1', '1', '0', '0', '0', '1', '1', '0', '0'],
            #                     ['0', '1', '0', '1', '0', '0', '1', '1', '0', '0'],
            #                     ['0', '0', '1', '1', '0', '0', '1', '1', '0', '0'],
            #                     ['0', '0', '0', '0', '0', '0', '1', '1', '0', '0'],
            #                     ['0', '1', '0', '0', '0', '0', '1', '1', '0', '0'],
            #                     ['0', '0', '1', '0', '0', '0', '1', '1', '0', '0'],
            #                     ['0', '0', '0', '1', '0', '0', '1', '1', '0', '0'],
            #                     ['0', '1', '1', '0', '0', '0', '1', '1', '0', '0'],
            #                     ['0', '0', '1', '1', '0', '0', '1', '1', '0', '0'],
            #                     ['0', '1', '0', '1', '0', '0', '1', '1', '0', '0']]

            for i in range(samples):
                candidate_label = str(random.randrange(0, 10))
                if candidate_label == '0':
                    f.write(','.join(candidate_data_0[random.randint(0, len(candidate_data_0)) - 1]) + "," + candidate_label + "\n")
                elif candidate_label == '1':
                    f.write(','.join(candidate_data_1[random.randint(0, len(candidate_data_1)) - 1]) + "," + candidate_label + "\n")
                elif candidate_label == '2':
                    f.write(','.join(candidate_data_2[random.randint(0, len(candidate_data_2)) - 1]) + "," + candidate_label + "\n")
                elif candidate_label == '3':
                    f.write(','.join(candidate_data_3[random.randint(0, len(candidate_data_3)) - 1]) + "," + candidate_label + "\n")
                elif candidate_label == '4':
                    f.write(','.join(candidate_data_4[random.randint(0, len(candidate_data_4)) - 1]) + "," + candidate_label + "\n")
                elif candidate_label == '5':
                    f.write(','.join(candidate_data_5[random.randint(0, len(candidate_data_5)) - 1]) + "," + candidate_label + "\n")
                elif candidate_label == '6':
                    f.write(','.join(candidate_data_6[random.randint(0, len(candidate_data_6)) - 1]) + "," + candidate_label + "\n")
                elif candidate_label == '7':
                    f.write(','.join(candidate_data_7[random.randint(0, len(candidate_data_7)) - 1]) + "," + candidate_label + "\n")
                elif candidate_label == '8':
                    f.write(','.join(candidate_data_8[random.randint(0, len(candidate_data_8)) - 1]) + "," + candidate_label + "\n")
                elif candidate_label == '9':
                    f.write(','.join(candidate_data_9[random.randint(0, len(candidate_data_9)) - 1]) + "," + candidate_label + "\n")

                # if candidate_label == '0':
                #     if can == 0:
                #         f.write(','.join(
                #             candidate_data_00[random.randint(0, len(candidate_data_00)) - 1]) + "," + candidate_label + "\n")

                #     elif can == 1:
                #         f.write(','.join(
                #             candidate_data_01[random.randint(0, len(candidate_data_01)) - 1]) + "," + candidate_label + "\n")

                #     elif can == 2:
                #         f.write(','.join(
                #             candidate_data_02[random.randint(0, len(candidate_data_02)) - 1]) + "," + candidate_label + "\n")

                # elif candidate_label == '1':
                #     if can == 0:
                #         f.write(','.join(candidate_data_10[random.randint(0, len(candidate_data_10)) - 1]) + "," + candidate_label + "\n")
                #     elif can == 1:
                #         f.write(','.join(candidate_data_11[random.randint(
                #             0, len(candidate_data_11)) - 1]) + "," + candidate_label + "\n")
                #     elif can == 2:
                #         f.write(','.join(candidate_data_12[random.randint(
                #             0, len(candidate_data_12)) - 1]) + "," + candidate_label + "\n")

                # elif candidate_label == '2':
                #     if can == 0:
                #         f.write(','.join(candidate_data_20[random.randint(
                #             0, len(candidate_data_20)) - 1]) + "," + candidate_label + "\n")
                #     elif can == 1::1

                #         f.write(','.join(candidate_data_21[random.randint(
                #             0, len(candidate_data_21)) - 1]) + "," + candidate_label + "\n")
                #     elif can == 2:
                #         f.write(','.join(candidate_data_12[random.randint(
                #             0, len(candidate_data_22)) - 1]) + "," + candidate_label + "\n")

        with open(os.path.join(NTE_MODULE_PATH, 'data','synth', 'blipmc', 'train.csv'), 'w') as f:
            print(int(samples * 0.8))
            gen(f, int(samples * 0.8))

        with open(os.path.join(NTE_MODULE_PATH, 'data', 'synth','blipmc', 'test.csv'), 'w') as f:
            print(samples - int(samples * 0.8))
            gen(f, samples - int(samples * 0.8))

        self.create_meta([os.path.join(NTE_MODULE_PATH, 'data', 'synth','blipmc', 'train.csv'), os.path.join(NTE_MODULE_PATH, 'data', 'synth','blipmc', 'test.csv')])


if __name__ == '__main__':
    BlipMCDataset()._generate(1000)
