import numpy as np


def salient_frequency(saliency, labels=None):
    if len(saliency.shape) > 1:
        if labels == None:
            labels = list(range(saliency.shape[0]))
        scores = {}

        for e, s in enumerate(saliency):
            # print("===="*20)
            # print(labels[e])
            # print(s)
            d = np.unique(np.digitize(s, bins=[np.min(s), np.max(s) / 2]), return_counts=True)
            # print(d)
            scores[labels[e]] = d[1][-1] / len(s),  round(float(np.var(s)),4)
        return scores
    else:
        d = np.unique(np.digitize(saliency, bins=[np.min(saliency), np.max(saliency) / 2]), return_counts=True)
    return d[1][-1] / len(saliency), round(float(np.var(saliency)),4)
