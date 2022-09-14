import torch
from nte import NTE_TRAINED_MODEL_PATH


class BurstExistenceDNNModel():
    def __new__(cls):
        return torch.load(NTE_TRAINED_MODEL_PATH + '/burst100/burst_existence_dnn_ce.ckpt')


class BurstExistenceRNNModel():
    def __new__(cls):
        return torch.load(NTE_TRAINED_MODEL_PATH + '/burst100/burst_existence_rnn_ce.ckpt')


class BurstLocationDNNModel():
    def __new__(cls):
        return torch.load(NTE_TRAINED_MODEL_PATH + '/burst100/burst_location_dnn_ce.ckpt')


class BurstLocationRNNModel():
    def __new__(cls):
        return torch.load(NTE_TRAINED_MODEL_PATH + '/burst100/burst_location_rnn_ce.ckpt')


class BurstFrequencyDNNModel():
    def __new__(cls):
        return torch.load(NTE_TRAINED_MODEL_PATH + '/burst100/burst_frequency_dnn_ce.ckpt')


class BurstFrequencyRNNModel():
    def __new__(cls):
        return torch.load(NTE_TRAINED_MODEL_PATH + '/burst100/burst_frequency_rnn_ce.ckpt')


class BurstStrengthDNNModel():
    def __new__(cls):
        return torch.load(NTE_TRAINED_MODEL_PATH + '/burst100/burst_strength_dnn_ce.ckpt')


class BurstStrengthRNNModel():
    def __new__(cls):
        return torch.load(NTE_TRAINED_MODEL_PATH + '/burst100/burst_strength_rnn_ce.ckpt')


class BurstTimeDifferenceStrengthDNNModel():
    def __new__(cls):
        return torch.load(NTE_TRAINED_MODEL_PATH + '/burst100/burst_time_difference_strength_dnn_ce.ckpt')


class BurstTimeDifferenceExistenceDNNModel():
    def __new__(cls):
        return torch.load(NTE_TRAINED_MODEL_PATH + '/burst100/burst_time_difference_existence_dnn_ce.ckpt')
