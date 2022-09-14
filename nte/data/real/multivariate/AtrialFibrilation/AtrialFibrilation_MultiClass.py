from nte.data.dataset import Dataset
from nte import NTE_MODULE_PATH
import pandas as pd
import os
import numpy as np

# http: // www.peterjbentley.com/heartchallenge/

class AtrialFibrilation_MultiClass(Dataset):
    def __init__(self):
        super().__init__(
            name='AtrialFibrilation',
            dims=2,
            meta_file_path=os.path.join(NTE_MODULE_PATH,'data','real','multivariate','AtrialFibrilation','meta.json'))

    def load_train_data(self):
        df1 = pd.read_csv(os.path.join(NTE_MODULE_PATH,'data','real','multivariate','AtrialFibrilation', 'train_d1.csv'))
        df2 = pd.read_csv(os.path.join(NTE_MODULE_PATH,'data','real','multivariate','AtrialFibrilation', 'train_d2.csv'))
        df=pd.concat([df1,df2,] ,join="inner",axis=1)
        df = df.loc[:,~df.columns.duplicated(keep='last')]
        df.loc[df['target']  == 'n', 'target'] = 0
        df.loc[df['target']  == 's', 'target'] = 1
        df.loc[df['target']  == 't', 'target'] = 2

        train_label = df[df.columns[-1]]
        train_label = train_label.values
        train_label = train_label.astype(int)
        train_data = df[df.columns[:-1]].to_numpy()
        print("columns are ")
        print(df.columns)
        return train_data, train_label

    def load_test_data(self):
        df1 = pd.read_csv(os.path.join(NTE_MODULE_PATH,'data','real','multivariate','AtrialFibrilation','test_d1.csv'))
        df2 = pd.read_csv(os.path.join(NTE_MODULE_PATH,'data','real','multivariate','AtrialFibrilation','test_d2.csv'))
        df=pd.concat([df1,df2,] ,join="inner",axis=1)
        df = df.loc[:,~df.columns.duplicated(keep='last')]
        df.loc[df['target']  == 'n', 'target'] = 0
        df.loc[df['target']  == 's', 'target'] = 1
        df.loc[df['target']  == 't', 'target'] = 2

        test_label = df[df.columns[-1]]
        test_label = test_label.values
        test_label = test_label.astype(int)
        test_data = df[df.columns[:-1]].to_numpy()
        return test_data, test_label
