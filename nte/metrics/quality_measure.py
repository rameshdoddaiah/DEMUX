from nte.metrics import Metric
import numpy as np
import enum
from sklearn.metrics import accuracy_score
import logging
from nte.utils import normalize
import operator
#matplotlib.use('Agg')
#matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
# from nte.data.blipv3.blipv3_dataset import BlipV3Dataset
from nte.data.real.univariate.binary.wafer import WaferDataset
import torch
import pandas as pd
import sys
import getopt

#from matplotlib import pyplot as plt


logger = logging.getLogger(__name__)



class QualityMeasureType(enum.Enum):
    TCR_RANDOM_NOSUB = enum.auto()
    TCR_ZERO_NOSUB = enum.auto()
    TC_ZERO_NOSUB = enum.auto()
    TC_INVERSE_NOSUB = enum.auto()
    TC_SWAP_SUB = enum.auto()
    TC_SWAP_SUB_ZERO = enum.auto()
    TC_SWAP_SUB_RANDOM = enum.auto()
    TC_SUB_MEAN = enum.auto()
    TC_SUB_MEAN_ZERO = enum.auto()
    TC_SUB_MEAN_RANDOM = enum.auto()

class QualityMeasure(Metric):

    def __init__(self, data, label, saliency, predict_fn, timesteps, metric_type='change'):
        """

        :param data:
        :param label:
        :param saliency:
        :param predict_fn:
        :param snrdb:
        :param accuracy_type: Supported 'change', 'mse'
        """
        super().__init__(data=data, saliency=saliency, label=label, predict_fn=predict_fn)
        self.probability_thresholds = [i / 100 for i in range(100)]
        self.metric_type = metric_type
        self.timesteps = timesteps
        self.cols  = list(range(self.timesteps))
        self.predictions = None

    def qm_tcr_random_generate_perturbation_no_subseq_only_input(self, inp):
        return inp * np.random.randint(2, size=self.timesteps)

    def qm_tcr_per_generate_perturbation_no_subseq_only_input(self, inp, saliency, masking='zero'):
        percentile_threshold = np.percentile(saliency, 90)
        s_max = np.amax(inp)
        r = np.random.randint(2, size=len(saliency))
        if masking == 'zero':
            return r * [0.0 if s >= percentile_threshold else inp[e] for e, s in enumerate(saliency)]
        elif masking == 'inverse':
            return r * [(s_max - inp[e]) * 1.0 if s >= percentile_threshold else inp[e] * 1.0 for e, s in
                        enumerate(saliency)]

    def qm_tc_per_generate_perturbation_no_subseq_only_input(self,inp, saliency, s_max, masking='zero'):
        percentile_threshold = np.percentile(saliency, 90)
        if masking == 'zero':
            return [0.0 if s >= percentile_threshold else inp[e] for e, s in enumerate(saliency)]
        elif masking == 'inverse':
            return [(s_max[e] - inp[e]) * 1.0 if s >= percentile_threshold else inp[e] * 1.0 for e, s in
                    enumerate(saliency)]

    def zero_tc_no_subseq(self):
        zero_tc_perturbation_values = pd.DataFrame(columns=self.cols)
        max_colwise = self.data.max(axis=0)
        for d,s in zip(self.data, self.saliency):
            pp = pd.Series(self.qm_tc_per_generate_perturbation_no_subseq_only_input(d,s, max_colwise, masking='zero'),
                           index=self.cols)
            zero_tc_perturbation_values = zero_tc_perturbation_values.append(pp, ignore_index=True)

        if self.predictions is None:
            _,self.predictions,_ = self.predict_fn(self.data)

        original_test_accuracy= accuracy_score(self.label, self.predictions) * 100
        logger.debug("QM(TC) Original Test data Accuracy No change in input subseq:        %.2f%%" % (original_test_accuracy))

        _,y_new_zero_tc_pred,_= self.predict_fn(zero_tc_perturbation_values.values)

        zero_tc_test_accuracy = accuracy_score(self.label, y_new_zero_tc_pred) * 100
        logger.debug("QM(TC) Zero Perturbed data Accuracy No change in input subseq:       %.2f%%" % (zero_tc_test_accuracy))
        return {"accuracy": original_test_accuracy, "perturbed_accuracy": zero_tc_test_accuracy,
                "type": QualityMeasureType.TC_ZERO_NOSUB.name}

    def inverse_tc_no_subseq(self):
        inverse_tc_perturbation_values = pd.DataFrame(columns=self.cols)
        max_colwise = self.data.max(axis=0)
        for d,s in zip(self.data, self.saliency):
            pp = pd.Series(self.qm_tc_per_generate_perturbation_no_subseq_only_input(d,s,max_colwise, masking='inverse'),
                           index=self.cols)
            inverse_tc_perturbation_values = inverse_tc_perturbation_values.append(pp, ignore_index=True)

        if self.predictions is None:
            _,self.predictions,_ = self.predict_fn(self.data)
        original_test_accuracy = accuracy_score(self.label, self.predictions) * 100.0
        logger.debug("QM(TC) Original Test data Accuracy No change in input subseq:        %.2f%%" % (original_test_accuracy))

        _,y_new_inverse_tc_pred,_ = self.predict_fn(inverse_tc_perturbation_values.values)

        inverse_tc_test_accuracy = accuracy_score(self.label, y_new_inverse_tc_pred) * 100.0
        logger.debug(
            "QM(TC) Inverse Perturbed data Accuracy No change in input subseq:    %.2f%%" % (inverse_tc_test_accuracy))

        return {"accuracy": original_test_accuracy, "perturbed_accuracy": inverse_tc_test_accuracy,
                "type": QualityMeasureType.TC_INVERSE_NOSUB.name}

    def random_tcr_no_subseq(self):
        random_tcr_perturbation_values = pd.DataFrame(columns=self.cols)
        for d in self.data:
            pp = pd.Series(self.qm_tcr_random_generate_perturbation_no_subseq_only_input(d), index=self.cols)
            random_tcr_perturbation_values = random_tcr_perturbation_values.append(pp, ignore_index=True)

        if self.predictions is None:
            _,self.predictions,_ = self.predict_fn(self.data)

        original_test_accuracy = accuracy_score(self.label, self.predictions) * 100.0
        logger.debug("QM(TCR) Original Test data Accuracy No change in input subseq:       %.2f%%" % (original_test_accuracy))
        _,y_new_random_tcr_pred,_ = self.predict_fn(random_tcr_perturbation_values.values.astype(np.float32))
        random_tcr_test_accuracy = accuracy_score(self.label, y_new_random_tcr_pred) * 100.0
        logger.debug(
            "QM(TCR) Random Perturbed data Accuracy No change in input subseq:    %.2f%%" % (random_tcr_test_accuracy))
        return {"accuracy":original_test_accuracy, "perturbed_accuracy":random_tcr_test_accuracy, "type":QualityMeasureType.TCR_RANDOM_NOSUB.name}

    def zero_tcr_no_subseq(self):
        zero_tcr_perturbation_values = pd.DataFrame(columns=self.cols)
        for d,s in zip(self.data, self.saliency):
            pp = pd.Series(
                self.qm_tcr_per_generate_perturbation_no_subseq_only_input(d, s,masking='zero'), index=self.cols)
            zero_tcr_perturbation_values = zero_tcr_perturbation_values.append(pp, ignore_index=True)

        if self.predictions is None:
            _,self.predictions,_ = self.predict_fn(self.data)

        original_test_accuracy = accuracy_score(self.label, self.predictions)
        logger.debug("QM(T) Original Test data Accuracy No change in input subseq:         %.2f%%" % (
                    original_test_accuracy * 100.0))

        _,y_new_zero_tcr_pred,_ = self.predict_fn(zero_tcr_perturbation_values.values)

        zero_tcr_test_accuracy = accuracy_score(self.label, y_new_zero_tcr_pred)
        logger.debug("QM(TCR) Zero Perturbed data Accuracy No chnage in input subseq:       %.2f%%" % (
                    zero_tcr_test_accuracy * 100.0))

        return {"accuracy": original_test_accuracy, "perturbed_accuracy": zero_tcr_test_accuracy,
         "type": QualityMeasureType.TCR_ZERO_NOSUB.name}

    def subsequence_helper(self, inp, saliency, qm_type:QualityMeasureType, win_size=1):
        percentile_threshold = np.percentile(saliency, 90)
        temp = np.zeros([len(inp)])
        skip_flag, skip_until = False, False
        seq_array = []
        si = 0
        for e, s in enumerate(saliency):
            if skip_flag == False:
                if s >= percentile_threshold:
                    skip_flag = True
                    skip_until = e + win_size - 1

                    if skip_until <= win_size:
                        skip_until = e + win_size

                    if e == 0:
                        e = 1

                    if skip_until > len(saliency):
                        skip_until = len(saliency)
                    if qm_type == QualityMeasureType.TC_SWAP_SUB_ZERO:
                        seq_array = inp[skip_until:e - 1:-1] * 0.0
                    elif qm_type == QualityMeasureType.TC_SWAP_SUB_RANDOM:
                        seq_array = inp[skip_until:e - 1:-1] * np.random.rand()
                    elif qm_type == QualityMeasureType.TC_SWAP_SUB:
                        seq_array = inp[skip_until:e - 1:-1]
                    elif qm_type == QualityMeasureType.TC_SUB_MEAN:
                        seq_mean = inp[e:skip_until].mean()
                    elif qm_type == QualityMeasureType.TC_SUB_MEAN_ZERO:
                        seq_mean = inp[e:skip_until].mean() *0.0
                    elif qm_type == QualityMeasureType.TC_SUB_MEAN_RANDOM:
                        seq_mean = inp[e:skip_until].mean() * np.random.rand()

                    if qm_type in [QualityMeasureType.TC_SUB_MEAN, QualityMeasureType.TC_SUB_MEAN_ZERO,
                                   QualityMeasureType.TC_SUB_MEAN_RANDOM]:
                        temp[e] = seq_mean
                    else:
                        si = 0
                        temp[e] = seq_array[si]
                        si += 1
                else:
                    temp[e] = inp[e]

            else:
                if qm_type in [QualityMeasureType.TC_SUB_MEAN, QualityMeasureType.TC_SUB_MEAN_ZERO,
                               QualityMeasureType.TC_SUB_MEAN_RANDOM]:
                    temp[e] = seq_mean
                else:
                    temp[e] = seq_array[si]
                    si += 1

                if e >= skip_until or si >= win_size:
                    skip_until = False
                    skip_flag = False
                    si = 0
                    seq_array = []
                    seq_mean = 0

        # todo Ramesh - Verify the type of randomness
        # r = np.random.randint(2, size=len(saliency))
        # temp = temp * r
        return temp

    def tc_sub_seq(self, qm_type:QualityMeasureType, **kwargs):
        if self.predictions is None:
            _, self.predictions, _ = self.predict_fn(self.data)
        original_test_accuracy = accuracy_score(self.label, self.predictions) * 100.0

        win_size_accuracy = {}
        if 'win_size' in kwargs:
            loop_range = [kwargs['win_size']]
        else:
            loop_range = range(2, self.timesteps)

        for win_size in loop_range:
            swap_tcs_perturbation_values = pd.DataFrame(columns=self.cols)
            for d,s in zip(self.data, self.saliency):
                pp = pd.Series(
                    self.subsequence_helper(d, s, qm_type, win_size), index=self.cols)
                swap_tcs_perturbation_values = swap_tcs_perturbation_values.append(pp, ignore_index=True)
            _,y_new_swap_tcs_pred,_ = self.predict_fn(swap_tcs_perturbation_values.values)
            swap_tcs_test_accuracy = accuracy_score(self.label, y_new_swap_tcs_pred)
            win_size_accuracy[win_size] = swap_tcs_test_accuracy * 100.0
            logger.debug("QM(TCR_SEQ) Win_size %d Swap Seq Perturbed data Accuracy: %.2f%%" % (
            win_size, swap_tcs_test_accuracy * 100.0))

        fig, ax = plt.subplots(1, 1, figsize=(11, 10), sharex=True)
        ax.plot((((list(win_size_accuracy.keys())))), list(win_size_accuracy.values()), marker='o', linestyle='-')
        ax.set_xlabel('Window Sizes')
        ax.set_ylabel('Accuracy')
        ax.set_title(f" {qm_type.name}: Subsequence length VS Accuracy")
        ax.set_xticks(range(len(win_size_accuracy)))
        #plt.show()
        return {"accuracy": original_test_accuracy, "perturbed_accuracy": win_size_accuracy,
                "type": qm_type.name}

    def evaluate(self, qm_type:QualityMeasureType, **kwargs):

        # TCR Measures
        if qm_type==QualityMeasureType.TCR_RANDOM_NOSUB:
            return self.random_tcr_no_subseq()

        elif qm_type == QualityMeasureType.TCR_ZERO_NOSUB:
            return self.zero_tcr_no_subseq()

        # TC Measures

        elif qm_type== QualityMeasureType.TC_ZERO_NOSUB:
            return self.zero_tc_no_subseq()

        elif qm_type== QualityMeasureType.TC_INVERSE_NOSUB:
            return self.inverse_tc_no_subseq()

        # TC Subsequence Swap Measures

        elif qm_type == QualityMeasureType.TC_SWAP_SUB:
            return self.tc_sub_seq(qm_type, **kwargs)

        elif qm_type == QualityMeasureType.TC_SWAP_SUB_ZERO:
            return self.tc_sub_seq(qm_type, **kwargs)

        elif qm_type == QualityMeasureType.TC_SWAP_SUB_RANDOM:
            return self.tc_sub_seq(qm_type, **kwargs)

        # TC Subsequence Mean Measures

        elif qm_type == QualityMeasureType.TC_SUB_MEAN:
            return self.tc_sub_seq(qm_type, **kwargs)

        elif qm_type == QualityMeasureType.TC_SUB_MEAN_ZERO:
            return self.tc_sub_seq(qm_type, **kwargs)

        elif qm_type == QualityMeasureType.TC_SUB_MEAN_RANDOM:
            return self.tc_sub_seq(qm_type, **kwargs)



def print_qme(qme):
    print("----"*25)
    if qme.get("type") == QualityMeasureType.TCR_RANDOM_NOSUB.name:
        #TCR Measures
        print("TCR Measures: ")
        print(f'Best {qme.get("type")} Accuracy {qme.get("perturbed_accuracy")}')
    elif qme.get("type") == QualityMeasureType.TC_ZERO_NOSUB.name:
        # TC Measures
        print("TC Measures: ")
        print(f'Best {qme.get("type")} Accuracy {qme.get("perturbed_accuracy")}')
    elif qme.get("type") == QualityMeasureType.TC_INVERSE_NOSUB.name:
        # TC Measures
        print("TC Measures: ")
        print(f'Best {qme.get("type")} Accuracy {qme.get("perturbed_accuracy")}')
    elif qme.get("type") == QualityMeasureType.TC_SWAP_SUB.name:
        # TC Subsequence Swap measures
        print(f'TC Subsequence {qme.get("type")} measures:')
        per_dict_acc = qme.get('perturbed_accuracy')
        print(f'Best Window Size {max(per_dict_acc.items(), key=operator.itemgetter(1))[0]}')
        print(f'Best Accuray Size {max(per_dict_acc.items(), key=operator.itemgetter(1))[1]}')
    elif qme.get("type") == QualityMeasureType.TC_SWAP_SUB_ZERO.name:
        # TC Subsequence Swap measures
        print(f'TC Subsequence {qme.get("type")} measures:')
        per_dict_acc = qme.get('perturbed_accuracy')
        print(f'Best Window Size {max(per_dict_acc.items(), key=operator.itemgetter(1))[0]}')
        print(f'Best Accuray Size {max(per_dict_acc.items(), key=operator.itemgetter(1))[1]}')
    elif qme.get("type") == QualityMeasureType.TC_SWAP_SUB_RANDOM.name:
        # TC Subsequence Swap measures
        print(f'TC Subsequence {qme.get("type")} measures:')
        per_dict_acc = qme.get('perturbed_accuracy')
        print(f'Best Window Size {max(per_dict_acc.items(), key=operator.itemgetter(1))[0]}')
        print(f'Best Accuray Size {max(per_dict_acc.items(), key=operator.itemgetter(1))[1]}')
    elif qme.get("type") == QualityMeasureType.TC_SUB_MEAN.name:
        # TC Subsequence Mean measures
        print(f'TC Subsequence {qme.get("type")} measures:')
        per_dict_acc = qme.get('perturbed_accuracy')
        print(f'Best Window Size {max(per_dict_acc.items(), key=operator.itemgetter(1))[0]}')
        print(f'Best Accuray Size {max(per_dict_acc.items(), key=operator.itemgetter(1))[1]}')
    elif qme.get("type") == QualityMeasureType.TC_SUB_MEAN_ZERO.name:
        # TC Subsequence Mean measures
        print(f'TC Subsequence {qme.get("type")} measures:')
        per_dict_acc = qme.get('perturbed_accuracy')
        print(f'Best Window Size {max(per_dict_acc.items(), key=operator.itemgetter(1))[0]}')
        print(f'Best Accuray Size {max(per_dict_acc.items(), key=operator.itemgetter(1))[1]}')
    elif qme.get("type") == QualityMeasureType.TC_SUB_MEAN_RANDOM.name:
        # TC Subsequence Mean measures
        print(f'TC Subsequence {qme.get("type")} measures:')
        per_dict_acc = qme.get('perturbed_accuracy')
        print(f'Best Window Size {max(per_dict_acc.items(), key=operator.itemgetter(1))[0]}')
        print(f'Best Accuray Size {max(per_dict_acc.items(), key=operator.itemgetter(1))[1]}')
    else:
        print(qme)

    print(qme)

def main(argv):
    # default paths
    bb_model_path = '../models/wafer/wafer_dnn_mse.ckpt'
    saliency_path = "../saliencies/wafer/wafer_dnn_mse_shap_saliency.csv"

    opts, args = getopt.getopt(argv, "dm:s:", ["model=../models/wafer/wafer_dnn_mse.ckpt", "saliency=../saliencies/wafer/wafer_dnn_mse_shap_saliency.csv"])
    for opt, arg in opts:
        if opt == '-h':
            print()
            sys.exit()
        elif opt in ("-d", "--dataset"):
            dataset = WaferDataset()
        elif opt in ("-m", "--model"):
            bb_model_path = arg
        elif opt in ("-s", "--saliency"):
            saliency_path = arg


    print(bb_model_path)
    print(saliency_path)
    #bb_model = torch.load('../models/blipv3/blip_v3_dnn_mse.ckpt')
    #dataset = BlipV3Dataset()
    #bb_model = torch.load('../models/wafer/wafer_dnn_mse.ckpt')
    bb_model = torch.load(bb_model_path)
    dataset = WaferDataset()
    #saliency = pd.read_csv("../saliencies/blipv3/blipv3_dnn_mse_shap_saliency.csv", header=0).values
    #saliency = pd.read_csv("../saliencies/blipv3/blipv3_dnn_mse_lime_saliency.csv", header=0).values
    # saliency = pd.read_csv("../saliencies/blipv3/blipv3_dnn_mse_l_saliency.csv", header=0).values
    # saliency = pd.read_csv("../saliencies/blipv3/blipv3_dnn_mse_c_saliency.csv", header=0).values
    #saliency = pd.read_csv("../saliencies/blipv3/blipv3_dnn_mse_gle_tuned_saliency.csv", header=0).values

    saliency = pd.read_csv(saliency_path, header=0).values
    # saliency = pd.read_csv("../saliencies/wafer/wafer_dnn_mse_lime_saliency.csv", header=0).values
    # saliency = pd.read_csv("../saliencies/wafer/wafer_dnn_mse_l_saliency.csv", header=0).values
    # saliency = pd.read_csv("../saliencies/wafer/wafer_dnn_mse_c_saliency.csv", header=0).values
    # saliency = pd.read_csv("../saliencies/wafer/wafer_dnn_mse_gle_tuned_saliency.csv", header=0).values

    saliency = [normalize(s) for s in saliency]

    qm_metric = QualityMeasure(data=dataset.test_data, label=dataset.test_label,
                                     saliency=saliency,
                                     predict_fn=bb_model.evaluate,
                                     timesteps=dataset.test_data.shape[1])
    print_qme(qm_metric.evaluate(qm_type=QualityMeasureType.TC_ZERO_NOSUB))
    print_qme(qm_metric.evaluate(qm_type=QualityMeasureType.TC_INVERSE_NOSUB))
    print_qme(qm_metric.evaluate(qm_type=QualityMeasureType.TCR_RANDOM_NOSUB))
    print_qme(qm_metric.evaluate(qm_type=QualityMeasureType.TC_SWAP_SUB))
    print_qme(qm_metric.evaluate(qm_type=QualityMeasureType.TC_SWAP_SUB_ZERO))
    print_qme(qm_metric.evaluate(qm_type=QualityMeasureType.TC_SWAP_SUB_RANDOM ))
    print_qme(qm_metric.evaluate(qm_type=QualityMeasureType.TC_SUB_MEAN))
    print_qme(qm_metric.evaluate(qm_type=QualityMeasureType.TC_SUB_MEAN_ZERO))
    print_qme(qm_metric.evaluate(qm_type=QualityMeasureType.TC_SUB_MEAN_RANDOM))

if __name__ == '__main__':
    main(sys.argv[1:])
#chart
