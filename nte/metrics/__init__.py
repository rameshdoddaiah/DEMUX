from abc import ABCMeta
from abc import abstractmethod
import numpy as np

class Metric(object, metaclass=ABCMeta):
    def __init__(self, data, label, saliency, predict_fn, **kwargs):
        self.score = None
        self.data = data
        self.label = label
        self.saliency = saliency
        self.predict_fn = predict_fn
        self.classes = np.unique(self.label)

    @abstractmethod
    def evaluate(self, **kwargs):
        pass

