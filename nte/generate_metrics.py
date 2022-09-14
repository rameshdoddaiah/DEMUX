from nte.saliencies.burst10 import *
from nte.trained_models.burst10 import *
from nte.metrics.quality_measure import QualityMeasure, QualityMeasureType, print_qme
import torch
from nte.metrics.insertion_and_delete import CausalMetric
import numpy as np
from nte.utils import Tee, normalize

SaliencyMethodOrder = ['Random', 'SHAP', 'LIME', 'L-S']

DATA_MODEL_SALIENCY = [
    [
        # BurstExistence10(),
        # BurstFrequency10(),
        BurstLocation10(),
        # BurstStrength10(),
        # BurstTimeDifferenceExistence10(),
        # BurstTimeDifferenceStrength10()
    ],
    [
        # [BurstExistenceDNNModel10(), BurstExistenceRNNModel10()],
        # [BurstFrequencyDNNModel10(), BurstFrequencyRNNModel10()],
        [BurstLocationDNNModel10(), BurstLocationRNNModel10()],
        # [BurstStrengthDNNModel10(), BurstStrengthRNNModel10()],
        # [BurstTimeDifferenceExistenceDNNModel10()],
        # [BurstTimeDifferenceStrengthDNNModel10()]
    ],
    [
        # [Burst_Random_Saliency(), BurstExistence_DNN_CE_SHAP_Saliency(), BurstExistence_DNN_CE_LIME_Saliency(),
        #  BurstExistence_DNN_CE_L_Saliency()],
        # [Burst_Random_Saliency(), BurstFrequency_DNN_CE_SHAP_Saliency(), BurstFrequency_DNN_CE_LIME_Saliency(),
        #  BurstFrequency_DNN_CE_L_Saliency()],
        [Burst_Random_Saliency(), BurstLocation_DNN_CE_SHAP_Saliency(), BurstLocation_DNN_CE_LIME_Saliency(),
         BurstLocation_DNN_CE_L_Saliency()],
        [Burst_Random_Saliency(), BurstLocation_RNN_CE_SHAP_Saliency(), BurstLocation_RNN_CE_LIME_Saliency(),
         BurstLocation_RNN_CE_L_Saliency()],
        # [Burst_Random_Saliency(), BurstStrength_DNN_CE_SHAP_Saliency(), BurstStrength_DNN_CE_LIME_Saliency(),
        #  BurstStrength_DNN_CE_L_Saliency()],
        # [Burst_Random_Saliency(), BurstTimeDifferenceExistence_DNN_CE_SHAP_Saliency(),
        #  BurstTimeDifferenceExistence_DNN_CE_LIME_Saliency(),
        #  BurstTimeDifferenceExistence_DNN_CE_L_Saliency()],
        # [Burst_Random_Saliency(), BurstTimeDifferenceStrength_DNN_CE_SHAP_Saliency(),
        #  BurstTimeDifferenceStrength_DNN_CE_LIME_Saliency(),
        #  BurstTimeDifferenceStrength_DNN_CE_L_Saliency()]
    ]
]


def calculate_qm(dataset, model, saliency, e):
    with Tee(f'/home/rdoddaiah/work/TimeSeriesSaliencyMaps/results/burst10/qm/{dataset.name}_{type(model)}_{SaliencyMethodOrder[e]}.log'):
        qm_metric = QualityMeasure(data=dataset.test_data, label=dataset.test_label,
                                   saliency=saliency,
                                   predict_fn=model.evaluate,
                                   timesteps=dataset.test_data.shape[1])
        print_qme(qm_metric.evaluate(qm_type=QualityMeasureType.TC_ZERO_NOSUB))
        print_qme(qm_metric.evaluate(qm_type=QualityMeasureType.TC_INVERSE_NOSUB))
        print_qme(qm_metric.evaluate(qm_type=QualityMeasureType.TCR_RANDOM_NOSUB))
        print_qme(qm_metric.evaluate(qm_type=QualityMeasureType.TC_SWAP_SUB))
        print_qme(qm_metric.evaluate(qm_type=QualityMeasureType.TC_SWAP_SUB_ZERO))
        print_qme(qm_metric.evaluate(qm_type=QualityMeasureType.TC_SWAP_SUB_RANDOM))
        print_qme(qm_metric.evaluate(qm_type=QualityMeasureType.TC_SUB_MEAN))
        print_qme(qm_metric.evaluate(qm_type=QualityMeasureType.TC_SUB_MEAN_ZERO))
        print_qme(qm_metric.evaluate(qm_type=QualityMeasureType.TC_SUB_MEAN_RANDOM))


def calculate_ins_del(dataset, model, saliency, e):
    with Tee(f'/home/rdoddaiah/work/TimeSeriesSaliencyMaps/results/burst10/ins_del/{dataset.name}_{type(model)}_{SaliencyMethodOrder[e]}_v123.log'):
        X_BATCH = dataset.test_data
        TIMESTEPS = X_BATCH.shape[1]
        noise = torch.tensor(np.random.normal(size=TIMESTEPS), dtype=torch.float32)
        blur = lambda x: x + noise
        insertion = CausalMetric(model, 'ins', 1, substrate_fn=blur)
        deletion = CausalMetric(model, 'del', 1, substrate_fn=torch.zeros_like)
        insertion.evaluate(time_series_tensor=torch.tensor(X_BATCH, dtype=torch.float32), exp_batch=saliency,
                           batch_size=8)
        deletion.evaluate(time_series_tensor=torch.tensor(X_BATCH, dtype=torch.float32), exp_batch=saliency,
                          batch_size=8)

with Tee(f'/home/rdoddaiah/work/TimeSeriesSaliencyMaps/results/burst10/ins_del/ins_del_dnn_V3.log'):
    for dataset, models, saliency_list in zip(DATA_MODEL_SALIENCY[0], DATA_MODEL_SALIENCY[1], DATA_MODEL_SALIENCY[2]):
        for model in models:
            # if isinstance(model, RNN):
            #     continue
            for e, saliency in enumerate(saliency_list):
                print(dataset.name, type(model), SaliencyMethodOrder[e])
                saliency = np.array([normalize(s) for s in saliency])
                calculate_ins_del(dataset, model, saliency, e)
                # calculate_qm(dataset, model, saliency, e)
