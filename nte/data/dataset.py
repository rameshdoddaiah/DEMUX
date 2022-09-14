import hashlib
import json
import datetime
from nte.utils import get_md5_checksum
from nte.utils.plot_utils import intialize_plot
import random
import numpy as np
import torch
import torch.utils.data as tdata
from fastdtw import fastdtw
from sklearn.metrics import euclidean_distances
import os
from abc import ABCMeta, abstractmethod
from nte.utils import CustomJsonEncoder
from sklearn.cluster import KMeans
from kneed import KneeLocator


class Dataset(tdata.Dataset, metaclass=ABCMeta):
    def __init__(self, name, meta_file_path, bb_model=None, cluster=True, dims=1):
        print("Loading train data . . .")
        self.train_data, self.train_label = self.load_train_data()
        print("Loading test data . . .")
        self.test_data, self.test_label = self.load_test_data()
        self.name = name
        self.meta_file_path = meta_file_path
        self.bb_model = bb_model
        self.dims=dims

        try:
            print(f"Loading meta file {self.meta_file_path}")
            self.meta = json.load(open(meta_file_path))
            # Load Train Summary Statistics
            self.num_classes = self.meta['num_classes']
            self.cluster_meta = self.meta["cluster_meta"]
            self.train_class_0_indices = self.meta['train_class_0_indices']
            self.train_class_1_indices = self.meta['train_class_1_indices']

            if self.num_classes > 2:
               self.train_class_2_indices = self.meta['train_class_2_indices']
               self.train_class_3_indices = self.meta['train_class_3_indices']
               self.train_class_4_indices = self.meta['train_class_4_indices']
               self.train_class_5_indices = self.meta['train_class_5_indices']
               self.train_class_6_indices = self.meta['train_class_6_indices']
               self.train_class_7_indices = self.meta['train_class_7_indices']
               self.train_class_8_indices = self.meta['train_class_8_indices']
               self.train_class_9_indices = self.meta['train_class_9_indices']
               self.train_class_10_indices = self.meta['train_class_10_indices']
               self.train_class_11_indices = self.meta['train_class_11_indices']

            self.train_class_0_data = np.take(self.train_data, self.train_class_0_indices, axis=0)
            self.train_class_1_data = np.take(self.train_data, self.train_class_1_indices, axis=0)

            if self.num_classes > 2:
               self.train_class_2_data = np.take(self.train_data, self.train_class_2_indices, axis=0)
               self.train_class_3_data = np.take(self.train_data, self.train_class_3_indices, axis=0)
               self.train_class_4_data = np.take(self.train_data, self.train_class_4_indices, axis=0)
               self.train_class_5_data = np.take(self.train_data, self.train_class_5_indices, axis=0)
               self.train_class_6_data = np.take(self.train_data, self.train_class_6_indices, axis=0)
               self.train_class_7_data = np.take(self.train_data, self.train_class_7_indices, axis=0)
               self.train_class_8_data = np.take(self.train_data, self.train_class_8_indices, axis=0)
               self.train_class_9_data = np.take(self.train_data, self.train_class_9_indices, axis=0)
               self.train_class_10_data = np.take(self.train_data, self.train_class_10_indices, axis=0)
               self.train_class_11_data = np.take(self.train_data, self.train_class_11_indices, axis=0)

            self.train_class_0_mean = np.array(self.meta['train_class_0_mean'])
            self.train_class_1_mean = np.array(self.meta['train_class_1_mean'])

            if self.num_classes > 2:
               self.train_class_2_mean = np.array(self.meta['train_class_2_mean'])
               self.train_class_3_mean = np.array(self.meta['train_class_3_mean'])
               self.train_class_4_mean = np.array(self.meta['train_class_4_mean'])
               self.train_class_5_mean = np.array(self.meta['train_class_5_mean'])
               self.train_class_6_mean = np.array(self.meta['train_class_6_mean'])
               self.train_class_7_mean = np.array(self.meta['train_class_7_mean'])
               self.train_class_8_mean = np.array(self.meta['train_class_8_mean'])
               self.train_class_9_mean = np.array(self.meta['train_class_9_mean'])
               self.train_class_10_mean = np.array(self.meta['train_class_10_mean'])
               self.train_class_11_mean = np.array(self.meta['train_class_11_mean'])

            # Load Test Summary Statistics
            self.test_class_0_indices = self.meta['test_class_0_indices']
            self.test_class_1_indices = self.meta['test_class_1_indices']

            if self.num_classes > 2:
               self.test_class_2_indices = self.meta['test_class_2_indices']
               self.test_class_3_indices = self.meta['test_class_3_indices']
               self.test_class_4_indices = self.meta['test_class_4_indices']
               self.test_class_5_indices = self.meta['test_class_5_indices']
               self.test_class_6_indices = self.meta['test_class_6_indices']
               self.test_class_7_indices = self.meta['test_class_7_indices']
               self.test_class_8_indices = self.meta['test_class_8_indices']
               self.test_class_9_indices = self.meta['test_class_9_indices']
               self.test_class_10_indices = self.meta['test_class_10_indices']
               self.test_class_11_indices = self.meta['test_class_11_indices']

            self.test_class_0_data = np.take(self.test_data, self.test_class_0_indices, axis=0)
            self.test_class_1_data = np.take(self.test_data, self.test_class_1_indices, axis=0)

            if self.num_classes > 2:
               self.test_class_2_data = np.take(self.test_data, self.test_class_2_indices, axis=0)
               self.test_class_3_data = np.take(self.test_data, self.test_class_3_indices, axis=0)
               self.test_class_4_data = np.take(self.test_data, self.test_class_4_indices, axis=0)
               self.test_class_5_data = np.take(self.test_data, self.test_class_5_indices, axis=0)
               self.test_class_6_data = np.take(self.test_data, self.test_class_6_indices, axis=0)
               self.test_class_7_data = np.take(self.test_data, self.test_class_7_indices, axis=0)
               self.test_class_8_data = np.take(self.test_data, self.test_class_8_indices, axis=0)
               self.test_class_9_data = np.take(self.test_data, self.test_class_9_indices, axis=0)
               self.test_class_10_data = np.take(self.test_data, self.test_class_10_indices, axis=0)
               self.test_class_11_data = np.take(self.test_data, self.test_class_11_indices, axis=0)

            self.test_class_0_mean = np.array(self.meta['test_class_0_mean'])
            self.test_class_1_mean = np.array(self.meta['test_class_1_mean'])

            if self.num_classes > 2:
               self.test_class_2_mean = np.array(self.meta['test_class_2_mean'])
               self.test_class_3_mean = np.array(self.meta['test_class_3_mean'])
               self.test_class_4_mean = np.array(self.meta['test_class_4_mean'])
               self.test_class_5_mean = np.array(self.meta['test_class_5_mean'])
               self.test_class_6_mean = np.array(self.meta['test_class_6_mean'])
               self.test_class_7_mean = np.array(self.meta['test_class_7_mean'])
               self.test_class_8_mean = np.array(self.meta['test_class_8_mean'])
               self.test_class_9_mean = np.array(self.meta['test_class_9_mean'])
               self.test_class_10_mean = np.array(self.meta['test_class_10_mean'])
               self.test_class_11_mean = np.array(self.meta['test_class_11_mean'])

            self.train_statistics = self.meta['train_statistics']
            self.test_statistics = self.meta['test_statistics']
            print("Meta file loaded successfully")
        except Exception as e:
            print("Meta file not found. Creating meta file . . .")
            # Clustering
            print("Creating cluster . . .")
            distorsions, kmeans_models, cluster_meta = [], [],  {}
            min_k, max_k = 2, 14
            print("Finding optimal value of K . . .")
            for k in range(min_k, max_k):
                kmeans_models.append(KMeans(n_clusters=k))
                kmeans_models[-1].fit(self.train_data)
                distorsions.append(kmeans_models[-1].inertia_)
            kn = KneeLocator(list(range(min_k, max_k)), distorsions, curve='convex', direction='decreasing')
            k_model = kmeans_models[kn.knee]
            print(f"Optimal value of K = {kn.knee}")
            cluster_idx = k_model.predict(self.train_data)
            cluster_meta = {
                "n_clusters": k_model.n_clusters,
                "centroids": k_model.cluster_centers_,
                **{f"cluster_{i}_indices": np.where(cluster_idx == i) for i in range(k_model.n_clusters)}
            }
            print("Cluster Meta completed!")

            # print(kn.knee)

            # Generate Train Summary Statistics
            self.num_classes = len(np.unique(self.train_label))
            self.train_class_0_indices = np.where(self.train_label == 0)[0]
            self.train_class_1_indices = np.where(self.train_label == 1)[0]

            if self.num_classes > 2:
               self.train_class_2_indices = np.where(self.train_label == 2)[0]
               self.train_class_3_indices = np.where(self.train_label == 3)[0]
               self.train_class_4_indices = np.where(self.train_label == 4)[0]
               self.train_class_5_indices = np.where(self.train_label == 5)[0]
               self.train_class_6_indices = np.where(self.train_label == 6)[0]
               self.train_class_7_indices = np.where(self.train_label == 7)[0]
               self.train_class_8_indices = np.where(self.train_label == 8)[0]
               self.train_class_9_indices = np.where(self.train_label == 9)[0]
               self.train_class_10_indices = np.where(self.train_label == 10)[0]
               self.train_class_11_indices = np.where(self.train_label == 11)[0]

            self.train_class_0_data = np.take(self.train_data, self.train_class_0_indices, axis=0)
            self.train_class_1_data = np.take(self.train_data, self.train_class_1_indices, axis=0)

            if self.num_classes > 2:
               self.train_class_2_data = np.take(self.train_data, self.train_class_2_indices, axis=0)
               self.train_class_3_data = np.take(self.train_data, self.train_class_3_indices, axis=0)
               self.train_class_4_data = np.take(self.train_data, self.train_class_4_indices, axis=0)
               self.train_class_5_data = np.take(self.train_data, self.train_class_5_indices, axis=0)
               self.train_class_6_data = np.take(self.train_data, self.train_class_6_indices, axis=0)
               self.train_class_7_data = np.take(self.train_data, self.train_class_7_indices, axis=0)
               self.train_class_8_data = np.take(self.train_data, self.train_class_8_indices, axis=0)
               self.train_class_9_data = np.take(self.train_data, self.train_class_9_indices, axis=0)
               self.train_class_10_data = np.take(self.train_data, self.train_class_10_indices, axis=0)
               self.train_class_11_data = np.take(self.train_data, self.train_class_11_indices, axis=0)

            self.train_class_0_mean = self.train_class_0_data.mean(axis=0)
            self.train_class_1_mean = self.train_class_1_data.mean(axis=0)

            if self.num_classes > 2:
               self.train_class_2_mean = self.train_class_2_data.mean(axis=0)
               self.train_class_3_mean = self.train_class_3_data.mean(axis=0)
               self.train_class_4_mean = self.train_class_4_data.mean(axis=0)
               self.train_class_5_mean = self.train_class_5_data.mean(axis=0)
               self.train_class_6_mean = self.train_class_6_data.mean(axis=0)
               self.train_class_7_mean = self.train_class_7_data.mean(axis=0)
               self.train_class_8_mean = self.train_class_8_data.mean(axis=0)
               self.train_class_9_mean = self.train_class_9_data.mean(axis=0)
               self.train_class_10_mean = self.train_class_10_data.mean(axis=0)
               self.train_class_11_mean = self.train_class_11_data.mean(axis=0)

            # Generate Test Summary Statistics
            self.test_class_0_indices = np.where(self.test_label == 0)[0]
            self.test_class_1_indices = np.where(self.test_label == 1)[0]

            if self.num_classes > 2:
               self.test_class_2_indices = np.where(self.test_label == 2)[0]
               self.test_class_3_indices = np.where(self.test_label == 3)[0]
               self.test_class_4_indices = np.where(self.test_label == 4)[0]
               self.test_class_5_indices = np.where(self.test_label == 5)[0]
               self.test_class_6_indices = np.where(self.test_label == 6)[0]
               self.test_class_7_indices = np.where(self.test_label == 7)[0]
               self.test_class_8_indices = np.where(self.test_label == 8)[0]
               self.test_class_9_indices = np.where(self.test_label == 9)[0]
               self.test_class_10_indices = np.where(self.test_label == 10)[0]
               self.test_class_11_indices = np.where(self.test_label == 11)[0]

            self.test_class_0_data = np.take(self.test_data, self.test_class_0_indices, axis=0)
            self.test_class_1_data = np.take(self.test_data, self.test_class_1_indices, axis=0)

            if self.num_classes > 2:
               self.test_class_2_data = np.take(self.test_data, self.test_class_2_indices, axis=0)
               self.test_class_3_data = np.take(self.test_data, self.test_class_3_indices, axis=0)
               self.test_class_4_data = np.take(self.test_data, self.test_class_4_indices, axis=0)
               self.test_class_5_data = np.take(self.test_data, self.test_class_5_indices, axis=0)
               self.test_class_6_data = np.take(self.test_data, self.test_class_6_indices, axis=0)
               self.test_class_7_data = np.take(self.test_data, self.test_class_7_indices, axis=0)
               self.test_class_8_data = np.take(self.test_data, self.test_class_8_indices, axis=0)
               self.test_class_9_data = np.take(self.test_data, self.test_class_9_indices, axis=0)
               self.test_class_10_data = np.take(self.test_data, self.test_class_10_indices, axis=0)
               self.test_class_11_data = np.take(self.test_data, self.test_class_11_indices, axis=0)

            self.test_class_0_mean = self.test_class_0_data.mean(axis=0)
            self.test_class_1_mean = self.test_class_1_data.mean(axis=0)

            if self.num_classes > 2:
               self.test_class_2_mean = self.test_class_2_data.mean(axis=0)
               self.test_class_3_mean = self.test_class_3_data.mean(axis=0)
               self.test_class_4_mean = self.test_class_4_data.mean(axis=0)
               self.test_class_5_mean = self.test_class_5_data.mean(axis=0)
               self.test_class_6_mean = self.test_class_6_data.mean(axis=0)
               self.test_class_7_mean = self.test_class_7_data.mean(axis=0)
               self.test_class_8_mean = self.test_class_8_data.mean(axis=0)
               self.test_class_9_mean = self.test_class_9_data.mean(axis=0)
               self.test_class_10_mean = self.test_class_10_data.mean(axis=0)
               self.test_class_11_mean = self.test_class_11_data.mean(axis=0)

            self.train_statistics = self.sample(dist_typ='dtw', data='train')
            self.test_statistics = self.sample(dist_typ='dtw', data='test')

            if self.num_classes > 2:
               self.meta = {
                   "cluster_meta":cluster_meta,
                   'num_classes': self.num_classes,
                   'train_class_0_indices': self.train_class_0_indices,
                   'train_class_1_indices': self.train_class_1_indices,
                   'train_class_2_indices': self.train_class_2_indices,
                   'train_class_3_indices': self.train_class_3_indices,
                   'train_class_4_indices': self.train_class_4_indices,
                   'train_class_5_indices': self.train_class_5_indices,
                   'train_class_6_indices': self.train_class_6_indices,
                   'train_class_7_indices': self.train_class_7_indices,
                   'train_class_8_indices': self.train_class_8_indices,
                   'train_class_9_indices': self.train_class_9_indices,
                   'train_class_10_indices': self.train_class_10_indices,
                   'train_class_11_indices': self.train_class_11_indices,
                   'train_class_0_mean': self.train_class_0_mean,
                   'train_class_1_mean': self.train_class_1_mean,
                   'train_class_2_mean': self.train_class_2_mean,
                   'train_class_3_mean': self.train_class_3_mean,
                   'train_class_4_mean': self.train_class_4_mean,
                   'train_class_5_mean': self.train_class_5_mean,
                   'train_class_6_mean': self.train_class_6_mean,
                   'train_class_7_mean': self.train_class_7_mean,
                   'train_class_8_mean': self.train_class_8_mean,
                   'train_class_9_mean': self.train_class_9_mean,
                   'train_class_10_mean': self.train_class_10_mean,
                   'train_class_11_mean': self.train_class_11_mean,
                   'test_class_0_indices': self.test_class_0_indices,
                   'test_class_1_indices': self.test_class_1_indices,
                   'test_class_2_indices': self.test_class_2_indices,
                   'test_class_3_indices': self.test_class_3_indices,
                   'test_class_4_indices': self.test_class_4_indices,
                   'test_class_5_indices': self.test_class_5_indices,
                   'test_class_6_indices': self.test_class_6_indices,
                   'test_class_7_indices': self.test_class_7_indices,
                   'test_class_8_indices': self.test_class_8_indices,
                   'test_class_9_indices': self.test_class_9_indices,
                   'test_class_10_indices': self.test_class_10_indices,
                   'test_class_11_indices': self.test_class_11_indices,
                   'test_class_0_mean': self.test_class_0_mean,
                   'test_class_1_mean': self.test_class_1_mean,
                   'test_class_2_mean': self.test_class_2_mean,
                   'test_class_3_mean': self.test_class_3_mean,
                   'test_class_4_mean': self.test_class_4_mean,
                   'test_class_5_mean': self.test_class_5_mean,
                   'test_class_6_mean': self.test_class_6_mean,
                   'test_class_7_mean': self.test_class_7_mean,
                   'test_class_8_mean': self.test_class_8_mean,
                   'test_class_9_mean': self.test_class_9_mean,
                   'test_class_10_mean': self.test_class_10_mean,
                   'test_class_11_mean': self.test_class_11_mean,
                   'train_statistics': self.train_statistics,
                   'test_statistics': self.test_statistics
               }
            else:
               self.meta = {
                   'train_class_0_indices': self.train_class_0_indices,
                   'train_class_1_indices': self.train_class_1_indices,
                   'train_class_0_mean': self.train_class_0_mean,
                   'train_class_1_mean': self.train_class_1_mean,
                   'test_class_0_indices': self.test_class_0_indices,
                   'test_class_1_indices': self.test_class_1_indices,
                   'test_class_0_mean': self.test_class_0_mean,
                   'test_class_1_mean': self.test_class_1_mean,
                   'train_statistics': self.train_statistics,
                   'test_statistics': self.test_statistics
               }

            with open(self.meta_file_path, 'w') as f:
                json.dump(self.meta, f, indent=2, cls=CustomJsonEncoder)
            print("Meta file created successfully")
        self.valid_data = []
        self.valid_label = []

        # self.train_meta = None
        # self.test_meta = None
        # self.valid_meta = None
        self.indices = {}

        # Prepare representatives
        self.representatives = self.representatives = {'train': self._generate_representatives('train'),
                                                       'test': self._generate_representatives('test')}
        self.valid_name = list(self.representatives['train'].keys())
        self.valid_data = np.array(self.valid_data)
        self.valid_label = np.array(self.valid_label)

    def _generate_representatives(self, data_type="train"):
        representatives = {}
        for k, v in self.meta[data_type + '_statistics']['between_class'].items():
            for e, vals in enumerate(v[:2]):
                representatives[f"between_class_{k}_class_{e}"] = vals
                if data_type == 'test':
                    self.valid_data.append(vals)
                    self.valid_label.append(float(e))

        for k, v in self.meta[data_type + '_statistics']['among_class_a'].items():
            for e, vals in enumerate(v[:2]):
                representatives[f"among_class_0_{k}_sample_{e}"] = vals
                if data_type == 'test':
                    self.valid_data.append(vals)
                    self.valid_label.append(0.0)

        for k, v in self.meta[data_type + '_statistics']['among_class_b'].items():
            for e, vals in enumerate(v[:2]):
                representatives[f"among_class_1_{k}_sample_{e}"] = vals
                if data_type == 'test':
                    self.valid_data.append(vals)
                    self.valid_label.append(1.0)

        for k, v in self.meta[data_type + '_statistics']['percentiles_a_data'].items():
            representatives[f"class_0_percentile_{k}"] = v[0]
            if data_type == 'test':
                self.valid_data.append(v[0])
                self.valid_label.append(0.0)

        for k, v in self.meta[data_type + '_statistics']['percentiles_b_data'].items():
            representatives[f"class_1_percentile_{k}"] = v[0]
            if data_type == 'test':
                self.valid_data.append(v[0])
                self.valid_label.append(1.0)
        return representatives

    def describe(self):
        stats = {
            'Timeseries Length': self.train_data.shape[1],
            'Train Samples': self.train_data.shape[0],
            'Test Samples': self.test_data.shape[0],
            'Train Event Rate': np.mean(self.train_label),
            'Test Event Rate': np.mean(self.test_label)
        }
        print(stats)
        return stats

    @abstractmethod
    def load_train_data(self) -> ():
        pass

    @abstractmethod
    def load_test_data(self) -> ():
        pass

    def _create_valid_from_train(self, valid_ratio: 0.2):
        val_index = self.train_data.shape[0] - int(self.train_data.shape[0] * valid_ratio)
        self.train_data, self.train_label = self.train_data[:val_index], self.train_label[:val_index]
        self.valid_data, self.valid_label = self.train_data[val_index:], self.train_label[val_index:]

    def __getitem__(self, ix):
        return self.train_data[ix], self.train_label[ix]

    def __len__(self):
        return len(self.train_data)

    def get_random_sample(self, cls=None):
        # todo Validation
        if cls is None:
            r = random.randrange(0, len(self.train_data))
            return self.train_data[r], self.train_label[r], r
        else:
            if cls not in self.indices:
                self.indices[cls] = np.where(self.train_label == cls)[0]
            r = random.randrange(0, len(self.indices[cls]))
            return self.train_data[self.indices[cls][r]], self.train_label[self.indices[cls][r]], self.indices[cls][
                r]

    def read_meta(self, file_list):
        meta_data = []
        for file in file_list:
            meta_data.append(json.load(open(file)))
        return meta_data

    def create_meta(self, file_list):
        md5checksum = get_md5_checksum(file_list)
        for file, md5 in zip(file_list, md5checksum):
            with open(file.split('.')[0] + '.meta', 'w') as f:
                json.dump({"md5": md5, "timestamp": str(datetime.datetime.now())}, f)

    def batch(self, batch_size=32):
        """
        Function to batch the data
        :param batch_size: batches
        :return: batches of X and Y
        """
        l = len(self.train_data)
        for ndx in range(0, l, batch_size):
            yield self.train_data[ndx:min(ndx + batch_size, l)], self.train_label[ndx:min(ndx + batch_size, l)]

    def sample(self, dist_typ='dtw', data='test', percentiles=[0.0, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0]):

        print(f"Computing representative samples for {data} using {dist_typ} distance")
        if dist_typ == 'euc':
            def dist_fn(a, b): return np.linalg.norm(a - b)
        elif dist_typ == 'dtw':
            def dist_fn(a, b): return fastdtw(a, b)[0]

        if data == 'test':
            class_0_data = self.test_class_0_data
            class_1_data = self.test_class_1_data
            class_0_indices = self.test_class_0_indices
            class_1_indices = self.test_class_1_indices
        elif data == 'train':
            class_0_data = self.train_class_0_data
            class_1_data = self.train_class_1_data
            class_0_indices = self.train_class_0_indices
            class_1_indices = self.train_class_1_indices

        # Between Classes - Max 2, Min 2
        # Among Classes - Max 2, Min 2

        samples = {'between_class': {'opposing': [], 'similar': []},
                   'among_class_a': {'opposing': [], 'similar': []},
                   'among_class_b': {'opposing': [], 'similar': []},
                   'percentiles_a': [], 'percentiles_b': []}
        # Get opposing classes
        min_dist, max_dist = float('inf'), 0.0

        # Between Class
        print("Computing between class samples . . .")
        for ea, point_a in enumerate(class_0_data):
            for eb, point_b in enumerate(class_1_data):
                dist = dist_fn(point_a.reshape([1, -1]), point_b.reshape([1, -1]))
                if dist < min_dist:
                    min_dist = dist
                    samples['between_class']['similar'] = (
                        point_a, point_b, min_dist, [class_0_indices[ea], class_1_indices[eb]])
                if dist > max_dist:
                    max_dist = dist
                    samples['between_class']['opposing'] = (
                        point_a, point_b, max_dist, [class_0_indices[ea], class_1_indices[eb]])

        # Among Class
        print("Computing among class 0 samples . . .")
        min_dist, max_dist = float('inf'), 0.0
        for ea, point_a in enumerate(class_0_data):
            for eb, point_b in enumerate(class_0_data):
                if ea != eb:
                    dist = dist_fn(point_a.reshape([1, -1]), point_b.reshape([1, -1]))
                    if dist < min_dist:
                        min_dist = dist
                        samples['among_class_a']['similar'] = (
                            point_a, point_b, min_dist, [class_0_indices[ea], class_0_data[eb]])
                    if dist > max_dist:
                        max_dist = dist
                        samples['among_class_a']['opposing'] = (
                            point_a, point_b, max_dist, [class_0_indices[ea], class_0_data[eb]])

        print("Computing among class 1 samples . . .")
        min_dist, max_dist = float('inf'), 0.0
        for ea, point_a in enumerate(class_1_data):
            for eb, point_b in enumerate(class_1_data):
                if ea != eb:
                    dist = dist_fn(point_a.reshape([1, -1]), point_b.reshape([1, -1]))
                    if dist < min_dist:
                        min_dist = dist
                        samples['among_class_b']['similar'] = (
                            point_a, point_b, min_dist, [class_1_indices[ea], class_1_indices[eb]])
                    if dist > max_dist:
                        max_dist = dist
                        samples['among_class_b']['opposing'] = (
                            point_a, point_b, max_dist, [class_1_indices[ea], class_1_indices[eb]])

        print("Computing percentiles . . .")
        # Get percentiles for each classes
        samples['percentiles_a'] = {percentiles[e]: q for e, q in
                                    enumerate(np.quantile(class_0_data, q=percentiles, axis=0))}
        samples['percentiles_b'] = {percentiles[e]: q for e, q in
                                    enumerate(np.quantile(class_1_data, q=percentiles, axis=0))}

        samples['percentiles_a_data'] = {q: [] for q, _ in samples['percentiles_a'].items()}
        samples['percentiles_b_data'] = {q: [] for q, _ in samples['percentiles_b'].items()}

        print("Matching percentiles for class 0 . . .")
        for q, percentile in samples['percentiles_a'].items():
            min_dist = float('inf')
            for point_a in class_0_data:
                dist = dist_fn(point_a, percentile)
                if dist < min_dist:
                    samples['percentiles_a_data'][q] = (point_a, dist)

        print("Matching percentiles for class 1 . . .")
        for q, percentile in samples['percentiles_b'].items():
            min_dist = float('inf')
            for point_b in class_1_data:
                dist = dist_fn(point_b, percentile)
                if dist < min_dist:
                    samples['percentiles_b_data'][q] = (point_b, dist)

        return samples

    def visualize(self, display=True):
        plt = intialize_plot()
        plt.figure(figsize=(15, 10))
        for i in range(5):
            ax = plt.subplot(int(f"32{i + 1}"), sharex=ax if i > 0 else None)
            d, l, idx = self.get_random_sample(cls=0)
            plt.plot(d, label=f"Idx: {idx} Class {l}")
            d, l, idx = self.get_random_sample(cls=1)
            plt.plot(d, label=f"Idx: {idx} Class {l}")
            plt.legend()
        plt.xlabel("Timesteps")
        plt.subplot(326)
        d = self.train_data[np.where(self.train_label == 0)].mean(0)
        plt.plot(d, label=f"Mean of Class {0}")
        d = self.train_data[np.where(self.train_label == 1)].mean(0)
        plt.plot(d, label=f"Mean of Class {1}")
        plt.xlabel("Timesteps")
        plt.legend()
        plt.suptitle(f"Summary of {self.name}", fontsize=18)
        if display:
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.show()
