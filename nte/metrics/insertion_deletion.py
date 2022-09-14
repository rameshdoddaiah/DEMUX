from nte.metrics import Metric
from nte.utils import generate_gaussian_noise
import numpy as np
from sklearn.metrics import accuracy_score, auc, roc_auc_score

class InsertionDeletionAUC(Metric):
    """
    Insertion and Deletion AUC - https://arxiv.org/pdf/1806.07421.pdf
    """
    def __init__(self, data, label, saliency, predict_fn, accuracy_type='accuracy'):
        """

        :param data:
        :param label:
        :param saliency:
        :param predict_fn:
        :param snrdb:
        :param accuracy_type: Supported 'accuracy', 'rocauc', 'gini'
        """
        super().__init__(data=data, saliency=saliency, label=label, predict_fn=predict_fn)
        self.probability_thresholds  = [i / 100 for i in range(100)]
        self.accuracy_type = accuracy_type


    def insertion_metric(self, snrdb):
        noise = generate_gaussian_noise(data=self.data, snrdb=snrdb)
        threshold_accuracies = {l: {} for l in self.classes}
        # Generate perturbed instances
        for p_threshold in self.probability_thresholds:
            perturbed_instances = []
            # Iterate over each candidate
            for d, n, saliency in zip(self.data, noise, self.saliency):
                perturbed_instances.append(np.array([d[e] if s <= p_threshold else d[e]*n[e] for e,s in enumerate(saliency)]))

            perturbed_instances = np.array(perturbed_instances)
            # Run the model
            _, predicted_class, prediction_probabilities = self.predict_fn(perturbed_instances)

            for cls in self.classes:
                class_labels_indicies = np.argwhere(self.label == cls)
                labels = self.label[class_labels_indicies].flatten()
                preds = predicted_class[class_labels_indicies].flatten()
                if self.accuracy_type =='accuracy':
                    threshold_accuracies[cls][p_threshold]= accuracy_score(y_true=labels, y_pred=preds)
                elif self.accuracy_type == 'rocauc':
                    threshold_accuracies[cls][p_threshold] = roc_auc_score(y_true=labels, y_score=preds)
                elif self.accuracy_type == 'gini':
                    threshold_accuracies[cls][p_threshold] = 2*roc_auc_score(y_true=labels, y_score=preds)-1
                else:
                    raise Exception(f'Unkonwn accuracy type: {self.accuracy_type}')

        class_auc = {}
        for cls, thresholds in threshold_accuracies.items():
            class_auc[cls]= auc(x=list(range(len(thresholds))), y=np.array(list(thresholds.values())))
        return class_auc, threshold_accuracies


    def deletion_metric(self, constant_value:float=0.0):
        threshold_accuracies = {l: {} for l in self.classes}
        # Generate perturbed instances
        for p_threshold in self.probability_thresholds:
            perturbed_instances = []
            # Iterate over each candidate
            for d, saliency, l in zip(self.data, self.saliency, self.label):
                perturbed_instances.append(
                    np.array([constant_value if s == p_threshold else constant_value for e, s in enumerate(saliency)]))

            perturbed_instances = np.array(perturbed_instances)
            # Run the model
            _, predicted_class, prediction_probabilities = self.predict_fn(perturbed_instances)
            for cls in self.classes:
                class_labels_indicies = np.argwhere(self.label==cls)
                labels = self.label[class_labels_indicies].flatten()
                preds = predicted_class[class_labels_indicies].flatten()
                if self.accuracy_type == 'accuracy':
                    threshold_accuracies[cls][p_threshold] = accuracy_score(y_true=labels, y_pred=preds)
                elif self.accuracy_type == 'rocauc':
                    threshold_accuracies[cls][p_threshold] = roc_auc_score(y_true=labels, y_score=preds)
                elif self.accuracy_type == 'gini':
                    threshold_accuracies[cls][p_threshold] = 2 * roc_auc_score(y_true=labels, y_score=preds) - 1
                else:
                    raise Exception(f'Unkonwn accuracy type: {self.accuracy_type}')
        class_auc = {}
        for cls, thresholds in threshold_accuracies.items():
            class_auc[cls] = auc(x=list(range(len(thresholds))), y=np.array(list(thresholds.values())))
            print(cls, thresholds)
        return class_auc, threshold_accuracies

    def evaluate(self, insertion_snrdb:float=20, deletion_constant_placement:float=0.0):
        insertion_auc, insertion_threshold_accuracies = self.insertion_metric(snrdb=insertion_snrdb)
        deletion_auc, deletion_threshold_accuracies = self.deletion_metric(constant_value=deletion_constant_placement)
        return insertion_auc, deletion_auc, insertion_threshold_accuracies, deletion_threshold_accuracies

if __name__ == '__main__':
    from nte.data.real.univariate.binary.wafer import WaferDataset
    import torch
    import pandas as pd
    from nte.utils import normalize

    # bb_model = torch.load('/Users/prathyushsp/Git/TimeSeriesSaliency/nte/models/blipv3/blip_v3_dnn_mse.ckpt')
    bb_model = torch.load('/Users/prathyushsp/Git/TimeSeriesSaliencyMaps/nte/models/wafer/wafer_dnn_mse.ckpt')

    # dataset = BlipV3Dataset()
    dataset = WaferDataset()
    # saliency = pd.read_csv("/Users/prathyushsp/Git/TimeSeriesSaliencyMaps/nte/saliencies/blipv3/blipv3_dnn_mse_lime_saliency.csv", header=0).values
    saliency = pd.read_csv(
        "/Users/prathyushsp/Git/TimeSeriesSaliencyMaps/nte/saliencies/wafer/wafer_dnn_mse_c_saliency.csv",
        header=0).values
    saliency = [normalize(d) for d in saliency]
    id_metric = InsertionDeletionAUC(data=dataset.test_data, label=dataset.test_label,
                                     saliency=saliency,#np.random.random(size=dataset.test_data.shape),
                                     predict_fn=bb_model.evaluate,
                                     accuracy_type='accuracy')
    ia, da, _, _ = id_metric.evaluate(insertion_snrdb=0.0)

    print(ia, da)



