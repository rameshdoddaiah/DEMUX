# from nte.data.blipv3.blipv3_dataset import BlipV4Dataset
from nte.models.saliency_model import SHAPSaliency, LimeSaliency
# from nte.models.blipv3.blip_v3_model import DNN_MSE_Model, RNN_MSE_Model
from nte import SALIENCY_PATH
from nte.models.saliency_model.l_saliency import LSaliency
import os
import numpy as np
import time
#from nte.trained_models.burst100 import *
from nte.trained_models.burst10 import *
from nte.utils import Tee


def gen_saliency(dataset, model, saliency_methods, config={}):
    for saliency_method in saliency_methods:
        saliency = saliency_method(data=dataset.test_data, label=dataset.test_label,
                                   model=model if str(saliency_method).split('.')[-1].replace("'>","") == "LSaliency" else model.evaluate,
                                   config=config)
        name = type(saliency).__name__
        with Tee(filename=os.path.join(SALIENCY_PATH, 'burst10', f'{dataset.name}_{type(model).__name__}_{name}.log')):
            print(f"Running {name} saliency . . .")
            start = time.time()
            saliency_value = saliency.generate_saliency(data=dataset.test_data, label=dataset.test_label)
            np.savetxt(os.path.join(SALIENCY_PATH, 'burst10', f'{dataset.name}_{type(model).__name__}_{name}.csv'),
                       saliency_value, delimiter=',',
                       header=",".join([f"s{i}" for i in range(dataset.test_data.shape[1])]), comments="")
            print(f"{name} saliency completed: Took {time.time() - start}s")


if __name__ == '__main__':
    datasets = [
        # BurstExistence10(),
        # BurstFrequency10(),
        BurstLocation10(),
        # BurstStrength10(),
        # BurstTimeDifferenceExistence10(),
        # BurstTimeDifferenceStrength10(),
    ]

    saliency_methods = [
        SHAPSaliency,
        LimeSaliency,
        LSaliency
    ]

    dnn_models = [
        # BurstExistenceDNNModel10(),
        # BurstFrequencyDNNModel10(),
        BurstLocationDNNModel10(),
        # BurstStrengthDNNModel10(),
        # BurstTimeDifferenceExistenceDNNModel10(),
        # BurstTimeDifferenceStrengthDNNModel10()
    ]

    rnn_models = [
        # BurstExistenceRNNModel10(),
        # BurstFrequencyRNNModel10(),
        BurstLocationRNNModel10(),
        # BurstStrengthRNNModel10(),
        # BurstTimeDifferenceExistenceRNNModel10(),
        # BurstTimeDifferenceStrengthRNNModel10()
    ]

    for dataset,d_model, r_model in zip(datasets, dnn_models, rnn_models):
        print(f"Working on Data: {dataset.name} | {type(d_model)}")
        from timeit import default_timer as timer
        start = timer()
        gen_saliency(dataset=dataset, model=d_model, saliency_methods=saliency_methods,
                     config={"timesteps": 10, "num_epochs": 1000, "learning_rate": 1e-3, "display_step": 100})
        gen_saliency(dataset=dataset, model=r_model, saliency_methods=saliency_methods,
                     config={"timesteps": 10, "num_epochs": 1000, "learning_rate": 1e-3, "display_step": 100})
        end = timer()
        print(end - start)

