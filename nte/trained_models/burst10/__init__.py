import torch
from nte import NTE_TRAINED_MODEL_PATH

class BurstExistenceDNNModel10():
    def __new__(cls):
        return torch.load(NTE_TRAINED_MODEL_PATH + '/burst10/burst_existence_dnn_ce.ckpt')


class BurstExistenceRNNModel10():
    def __new__(cls):
        return torch.load(NTE_TRAINED_MODEL_PATH + '/burst10/burst_existence_rnn_ce.ckpt')


class BurstLocationDNNModel10():
    def __new__(cls):
        return torch.load(NTE_TRAINED_MODEL_PATH + '/burst10/burst_location_dnn_ce.ckpt')


class BurstLocationRNNModel10():
    def __new__(cls):
        return torch.load(NTE_TRAINED_MODEL_PATH + '/burst10/burst_location_rnn_ce.ckpt')


class BurstFrequencyDNNModel10():
    def __new__(cls):
        return torch.load(NTE_TRAINED_MODEL_PATH + '/burst10/burst_frequency_dnn_ce.ckpt')


class BurstFrequencyRNNModel10():
    def __new__(cls):
        return torch.load(NTE_TRAINED_MODEL_PATH + '/burst10/burst_frequency_rnn_ce.ckpt')


class BurstStrengthDNNModel10():
    def __new__(cls):
        return torch.load(NTE_TRAINED_MODEL_PATH + '/burst10/burst_strength_dnn_ce.ckpt')


class BurstStrengthRNNModel10():
    def __new__(cls):
        return torch.load(NTE_TRAINED_MODEL_PATH + '/burst10/burst_strength_rnn_ce.ckpt')


class BurstTimeDifferenceStrengthDNNModel10():
    def __new__(cls):
        return torch.load(NTE_TRAINED_MODEL_PATH + '/burst10/burst_time_difference_time_difference_strength_dnn_ce.ckpt')


class BurstTimeDifferenceExistenceDNNModel10():
    def __new__(cls):
        return torch.load(NTE_TRAINED_MODEL_PATH + '/burst10/burst_time_difference_existence_dnn_ce.ckpt')

class BurstTimeDifferenceStrengthRNNModel10():
    def __new__(cls):
        return torch.load(NTE_TRAINED_MODEL_PATH + '/burst10/burst_time_difference_time_difference_strength_rnn_ce.ckpt')

class BurstTimeDifferenceExistenceRNNModel10():
    def __new__(cls):
        return torch.load(NTE_TRAINED_MODEL_PATH + '/burst10/burst_time_difference_existence_rnn_ce.ckpt')
