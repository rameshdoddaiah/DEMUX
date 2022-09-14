import pandas as pd
from nte import NTE_SALIENCY_PATH
import numpy as np


class Burst_Random_Saliency():
    def __new__(cls, *args, **kwargs):
        saliencies = pd.read_csv(NTE_SALIENCY_PATH + "/burst10/burst_location_Linear_SHAPSaliency.csv", header=0)
        return np.random.uniform(0, 1, size=saliencies.values.shape)


class BurstExistence_DNN_CE_SHAP_Saliency():
    def __new__(cls, *args, **kwargs):
        saliencies = pd.read_csv(NTE_SALIENCY_PATH + "/burst10/burst_existence_Linear_SHAPSaliency.csv", header=0)
        return saliencies.values


class BurstExistence_DNN_CE_LIME_Saliency():
    def __new__(cls, *args, **kwargs):
        saliencies = pd.read_csv(NTE_SALIENCY_PATH + "/burst10/burst_existence_Linear_LimeSaliency.csv", header=0)
        return saliencies.values


class BurstExistence_DNN_CE_L_Saliency():
    def __new__(cls, *args, **kwargs):
        saliencies = pd.read_csv(NTE_SALIENCY_PATH + "/burst10/burst_existence_Linear_LSaliency.csv", header=0)
        return saliencies.values


class BurstLocation_DNN_CE_SHAP_Saliency():
    def __new__(cls, *args, **kwargs):
        saliencies = pd.read_csv(NTE_SALIENCY_PATH + "/burst10/burst_location_Linear_SHAPSaliency.csv", header=0)
        return saliencies.values


class BurstLocation_DNN_CE_LIME_Saliency():
    def __new__(cls, *args, **kwargs):
        saliencies = pd.read_csv(NTE_SALIENCY_PATH + "/burst10/burst_location_Linear_LimeSaliency.csv", header=0)
        return saliencies.values


class BurstLocation_DNN_CE_L_Saliency():
    def __new__(cls, *args, **kwargs):
        saliencies = pd.read_csv(NTE_SALIENCY_PATH + "/burst10/burst_location_Linear_LSaliency.csv", header=0)
        return saliencies.values


class BurstFrequency_DNN_CE_SHAP_Saliency():
    def __new__(cls, *args, **kwargs):
        saliencies = pd.read_csv(NTE_SALIENCY_PATH + "/burst10/burst_frequency_Linear_SHAPSaliency.csv", header=0)
        return saliencies.values


class BurstFrequency_DNN_CE_LIME_Saliency():
    def __new__(cls, *args, **kwargs):
        saliencies = pd.read_csv(NTE_SALIENCY_PATH + "/burst10/burst_frequency_Linear_LimeSaliency.csv", header=0)
        return saliencies.values


class BurstFrequency_DNN_CE_L_Saliency():
    def __new__(cls, *args, **kwargs):
        saliencies = pd.read_csv(NTE_SALIENCY_PATH + "/burst10/burst_frequency_Linear_LSaliency.csv", header=0)
        return saliencies.values


class BurstStrength_DNN_CE_SHAP_Saliency():
    def __new__(cls, *args, **kwargs):
        saliencies = pd.read_csv(NTE_SALIENCY_PATH + "/burst10/burst_strength_Linear_SHAPSaliency.csv", header=0)
        return saliencies.values


class BurstStrength_DNN_CE_LIME_Saliency():
    def __new__(cls, *args, **kwargs):
        saliencies = pd.read_csv(NTE_SALIENCY_PATH + "/burst10/burst_strength_Linear_LimeSaliency.csv", header=0)
        return saliencies.values


class BurstStrength_DNN_CE_L_Saliency():
    def __new__(cls, *args, **kwargs):
        saliencies = pd.read_csv(NTE_SALIENCY_PATH + "/burst10/burst_strength_Linear_LSaliency.csv", header=0)
        return saliencies.values


"""
RNN

"""


class BurstExistence_RNN_CE_SHAP_Saliency():
    def __new__(cls, *args, **kwargs):
        saliencies = pd.read_csv(NTE_SALIENCY_PATH + "/burst10/burst_existence_RNN_SHAPSaliency.csv", header=0)
        return saliencies.values


class BurstExistence_RNN_CE_LIME_Saliency():
    def __new__(cls, *args, **kwargs):
        saliencies = pd.read_csv(NTE_SALIENCY_PATH + "/burst10/burst_existence_RNN_LimeSaliency.csv", header=0)
        return saliencies.values


class BurstExistence_RNN_CE_L_Saliency():
    def __new__(cls, *args, **kwargs):
        saliencies = pd.read_csv(NTE_SALIENCY_PATH + "/burst10/burst_existence_RNN_LSaliency.csv", header=0)
        return saliencies.values


class BurstLocation_RNN_CE_SHAP_Saliency():
    def __new__(cls, *args, **kwargs):
        saliencies = pd.read_csv(NTE_SALIENCY_PATH + "/burst10/burst_location_RNN_SHAPSaliency.csv", header=0)
        return saliencies.values


class BurstLocation_RNN_CE_LIME_Saliency():
    def __new__(cls, *args, **kwargs):
        saliencies = pd.read_csv(NTE_SALIENCY_PATH + "/burst10/burst_location_RNN_LimeSaliency.csv", header=0)
        return saliencies.values


class BurstLocation_RNN_CE_L_Saliency():
    def __new__(cls, *args, **kwargs):
        saliencies = pd.read_csv(NTE_SALIENCY_PATH + "/burst10/burst_location_RNN_LSaliency.csv", header=0)
        return saliencies.values


class BurstFrequency_RNN_CE_SHAP_Saliency():
    def __new__(cls, *args, **kwargs):
        saliencies = pd.read_csv(NTE_SALIENCY_PATH + "/burst10/burst_frequency_RNN_SHAPSaliency.csv", header=0)
        return saliencies.values


class BurstFrequency_RNN_CE_LIME_Saliency():
    def __new__(cls, *args, **kwargs):
        saliencies = pd.read_csv(NTE_SALIENCY_PATH + "/burst10/burst_frequency_RNN_LimeSaliency.csv", header=0)
        return saliencies.values


class BurstFrequency_RNN_CE_L_Saliency():
    def __new__(cls, *args, **kwargs):
        saliencies = pd.read_csv(NTE_SALIENCY_PATH + "/burst10/burst_frequency_RNN_LSaliency.csv", header=0)
        return saliencies.values


class BurstStrength_RNN_CE_SHAP_Saliency():
    def __new__(cls, *args, **kwargs):
        saliencies = pd.read_csv(NTE_SALIENCY_PATH + "/burst10/burst_strength_RNN_SHAPSaliency.csv", header=0)
        return saliencies.values


class BurstStrength_RNN_CE_LIME_Saliency():
    def __new__(cls, *args, **kwargs):
        saliencies = pd.read_csv(NTE_SALIENCY_PATH + "/burst10/burst_strength_RNN_LimeSaliency.csv", header=0)
        return saliencies.values


class BurstStrength_RNN_CE_L_Saliency():
    def __new__(cls, *args, **kwargs):
        saliencies = pd.read_csv(NTE_SALIENCY_PATH + "/burst10/burst_strength_RNN_LSaliency.csv", header=0)
        return saliencies.values



class BurstTimeDifferenceExistence_DNN_CE_SHAP_Saliency():
    def __new__(cls, *args, **kwargs):
        saliencies = pd.read_csv(NTE_SALIENCY_PATH + "/burst10/burst_time_difference_existence_Linear_SHAPSaliency.csv", header=0)
        return saliencies.values

class BurstTimeDifferenceExistence_DNN_CE_LIME_Saliency():
    def __new__(cls, *args, **kwargs):
        saliencies = pd.read_csv(NTE_SALIENCY_PATH + "/burst10/burst_time_difference_existence_Linear_LimeSaliency.csv", header=0)
        return saliencies.values

class BurstTimeDifferenceExistence_DNN_CE_L_Saliency():
    def __new__(cls, *args, **kwargs):
        saliencies = pd.read_csv(NTE_SALIENCY_PATH + "/burst10/burst_time_difference_existence_Linear_LSaliency.csv", header=0)
        return saliencies.values

class BurstTimeDifferenceExistence_RNN_CE_SHAP_Saliency():
    def __new__(cls, *args, **kwargs):
        saliencies = pd.read_csv(NTE_SALIENCY_PATH + "/burst10/burst_time_difference_existence_RNN_SHAPSaliency.csv", header=0)
        return saliencies.values

class BurstTimeDifferenceExistence_RNN_CE_LIME_Saliency():
    def __new__(cls, *args, **kwargs):
        saliencies = pd.read_csv(NTE_SALIENCY_PATH + "/burst10/burst_time_difference_existence_RNN_LimeSaliency.csv", header=0)
        return saliencies.values

class BurstTimeDifferenceExistence_RNN_CE_L_Saliency():
    def __new__(cls, *args, **kwargs):
        saliencies = pd.read_csv(NTE_SALIENCY_PATH + "/burst10/burst_time_difference_existence_RNN_LSaliency.csv", header=0)
        return saliencies.values


class BurstTimeDifferenceStrength_DNN_CE_SHAP_Saliency():
    def __new__(cls, *args, **kwargs):
        saliencies = pd.read_csv(NTE_SALIENCY_PATH + "/burst10/burst_time_difference_time_difference_strength_Linear_SHAPSaliency.csv", header=0)
        return saliencies.values

class BurstTimeDifferenceStrength_DNN_CE_LIME_Saliency():
    def __new__(cls, *args, **kwargs):
        saliencies = pd.read_csv(NTE_SALIENCY_PATH + "/burst10/burst_time_difference_time_difference_strength_Linear_LimeSaliency.csv", header=0)
        return saliencies.values

class BurstTimeDifferenceStrength_DNN_CE_L_Saliency():
    def __new__(cls, *args, **kwargs):
        saliencies = pd.read_csv(NTE_SALIENCY_PATH + "/burst10/burst_time_difference_time_difference_strength_Linear_LSaliency.csv", header=0)
        return saliencies.values



class BurstTimeDifferenceStrength_RNN_CE_SHAP_Saliency():
    def __new__(cls, *args, **kwargs):
        saliencies = pd.read_csv(NTE_SALIENCY_PATH + "/burst10/burst_time_difference_time_difference_strength_RNN_SHAPSaliency.csv", header=0)
        return saliencies.values

class BurstTimeDifferenceStrength_RNN_CE_LIME_Saliency():
    def __new__(cls, *args, **kwargs):
        saliencies = pd.read_csv(NTE_SALIENCY_PATH + "/burst10/burst_time_difference_time_difference_strength_RNN_LimeSaliency.csv", header=0)
        return saliencies.values

class BurstTimeDifferenceStrength_RNN_CE_L_Saliency():
    def __new__(cls, *args, **kwargs):
        saliencies = pd.read_csv(NTE_SALIENCY_PATH + "/burst10/burst_time_difference_time_difference_strength_RNN_LSaliency.csv", header=0)
        return saliencies.values



