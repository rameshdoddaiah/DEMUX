from nte.metrics import Metric
import numpy as np


class MeanPercentageChange(Metric):

    def __init__(self, data, label, saliency, predict_fn, metric_type='change'):
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

    def evaluate(self, constant_value_placement:float=0.0, **kwargs):

        threshold_accuracies = {}

        predicted_value, predicted_class, prediction_probabilities = self.predict_fn(self.data)
        # Generate perturbed instances
        for p_threshold in self.probability_thresholds:
            perturbed_instances = []
            # Iterate over each candidate
            for d, saliency in zip(self.data, self.saliency):
                perturbed_instances.append(
                    np.array([d[e] if s <= p_threshold else constant_value_placement for e, s in enumerate(saliency)]))

            perturbed_instances = np.array(perturbed_instances)
            # Run the model
            perturbed_predicted_value, perturbed_predicted_class, perturbed_prediction_probabilities = self.predict_fn(perturbed_instances)

            if self.metric_type == 'change':
                threshold_accuracies[p_threshold] = np.mean([t[c]-p[c] for t,p,c in zip(prediction_probabilities, perturbed_prediction_probabilities, predicted_class)])
            elif self.metric_type == 'mse':
                threshold_accuracies[p_threshold] = np.mean(np.square(prediction_probabilities - perturbed_prediction_probabilities))
            else:
                raise Exception(f"Unknown Metric Type: {self.metric_type}")

        return max(threshold_accuracies.items(), key = lambda k : k[1]),  threshold_accuracies



if __name__ == '__main__':
    from nte.data.synth.blipv3.blipv3_dataset import BlipV3Dataset
    import torch
    import pandas as pd

    bb_model = torch.load('/Users/prathyushsp/Git/TimeSeriesSaliency/nte/models/blipv3/blip_v3_dnn_mse.ckpt')


    dataset = BlipV3Dataset()
    saliency = pd.read_csv("/Users/prathyushsp/Git/TimeSeriesSaliency/saliencies/learned_masking/saliencymodel_blipv3_dnn_zerosinit_class0_mse_saliency.csv", header=0).values
    id_metric = MeanPercentageChange(data=dataset.test_data, label=dataset.test_label,
                                     saliency=saliency,#np.random.random(size=dataset.test_data.shape),
                                     predict_fn=bb_model.evaluate,
                                     metric_type='change')
    a, b= id_metric.evaluate()

    print(a)
    # print(b)