# -*- coding: utf-8 -*-
"""
| **@created on:** 9/23/20,
| **@author:** prathyushsp,
| **@version:** v0.0.1
|
| **Description:**
| 
|
| **Sphinx Documentation Status:** 
"""

from fastdtw import fastdtw
from scipy.spatial.distance import euclidean


class PerturbationManager(object):
    def __init__(self, original_signal, algo, prediction_prob, original_label, sample_id):
        self.original_signal = original_signal.flatten()
        self.sample_id = sample_id

        # self.absolute_euclidean = []
        # self.absolute_dtw = []
        # self.relative_euclidean = []
        # self.relative_dtw = []
        # self.step = []
        self.algo = algo
        self.prediction_prob = prediction_prob
        self.original_label = original_label
        self.perturbations = [original_signal]
        # self.confidence = []
        self.rows = []
        self.column_names = []
        # self.final_auc = []
        # self.insertion = []
        # self.deletion = []

    def add_perturbation(self, perturbation, step, confidence, saliency,
                         **kwargs):  # final_auc=0.0, insertion=0.0, deletion=0.0):

        weighted_euc_dist = lambda x, y: euclidean(x, y, w=saliency)

        if not self.column_names:
            self.column_names = [*[f'f{str(i)}' for i in range(len(self.original_signal))],
                                 *[f's{str(i)}' for i in range(len(self.original_signal))],
                                 "type", "sample_id", "algo", "itr", "label", "prob",
                                 "abs_euc", "abs_dtw", "rel_euc", "rel_dtw",
                                 "w_rel_euc", "w_rel_dtw", "w_abs_euc", "w_abs_dtw",
                                 *list(kwargs.keys())]
            self.rows.append(
                [*self.original_signal.tolist(), *[f"{1}" for _ in self.original_signal.tolist()], "o", self.sample_id,
                 self.algo, step, self.original_label, confidence,
                 "0", "0", "0", "0","0", "0", "0", "0", *["0" for _ in kwargs.keys()]])
        step += 1
        perturbation = perturbation.flatten()
        self.perturbations.append(perturbation)
        row = [*perturbation.tolist()]
        row = [*row, *saliency.tolist()]
        row = [*row, *["p", self.sample_id, self.algo, step, self.original_label, confidence]]
        # self.absolute_euclidean.append(euclidean(self.original_signal, perturbation))
        row.append(euclidean(self.original_signal, perturbation))
        # self.absolute_dtw.append(fastdtw(self.original_signal, perturbation, dist=euclidean))
        row.append(fastdtw(self.original_signal, perturbation)[0])
        if len(self.rows) >= 1:
            # self.relative_euclidean.append(euclidean(self.perturbations[-1], perturbation))
            row.append(euclidean(self.perturbations[-2], self.perturbations[-1]))
            # self.relative_dtw.append(fastdtw(self.perturbations[-1], perturbation, dist=euclidean))
            row.append(fastdtw(self.perturbations[-2], self.perturbations[-1], dist=euclidean)[0])

            row.append(euclidean(self.perturbations[-2], self.perturbations[-1], w=saliency))
            row.append(fastdtw(self.perturbations[-2], self.perturbations[-1], dist=weighted_euc_dist)[0])
        else:
            # self.relative_euclidean.append(0)
            row.append(0)
            # self.relative_dtw.append([0])
            row.append(0)
            row.append(0)
            row.append(0)

        row.append(euclidean(self.original_signal, perturbation, w=saliency))
        row.append(fastdtw(self.original_signal, perturbation, dist=weighted_euc_dist)[0])

        row = [*row, *[float(v) for v in kwargs.values()]]
        # self.step.append(step)
        # self.metrics.append(list(kwargs[]))
        # self.confidence.append(confidence)
        # self.final_auc.append(final_auc)
        # self.insertion.append(insertion)
        # self.deletion.append(deletion)
        self.rows.append(row)

    def update_perturbation(self, perturbations, confidences):
        for e, (perturbation, c) in enumerate(zip(perturbations, confidences)):
            self.absolute_euclidean.append(euclidean(self.original_signal, perturbation))
            self.absolute_dtw.append(fastdtw(self.original_signal, perturbation, dist=euclidean))
            if len(self.perturbations) >= 1:
                self.relative_euclidean.append(euclidean(self.perturbations[-1], perturbation))
                self.relative_dtw.append(fastdtw(self.perturbations[-1], perturbation, dist=euclidean))
            else:
                self.relative_euclidean.append(0)
                self.relative_dtw.append([0])
            self.perturbations.append(perturbation)
            self.step.append(e)
            self.confidence.append(c)

    # def to_csv(self, SAVE_DIR, TAG, UUID):
    #     save_path = f"{SAVE_DIR}/perturbations-{TAG}-{UUID}.csv"
    #     print(f"Saving perturbations to {save_path} . . .")
    #     with open(save_path, "w") as f:
    #         f.write(",".join([*[f'f{str(i)}' for i in range(len(self.original_signal))],
    #                           "type", "index", "algo", "itr", "abs_euc", "abs_dtw", "rel_euc", "rel_dtw", "prob",
    #                           "label", "final_auc", "insertion", "deletion"]) + "\n")
    #         f.write(",".join([*[f'{str(i)}' for i in self.original_signal],
    #                           "o", "-1", f"{self.algo}", "0", "0", "0", "0", "0", f"{self.prediction_prob}",
    #                           f"{self.original_label}", "0", "0", "0"]) + "\n")
    #         for e, (perturbation, itr) in enumerate(zip(self.perturbations, self.step)):
    #             f.write(",".join([*[f'{str(i)}' for i in perturbation],
    #                               "p", f"{e}", f"{self.algo}", f"{itr}",
    #                               f"{self.absolute_euclidean[e]}", f"{self.relative_euclidean[e]}",
    #                               f"{self.absolute_dtw[e][0]}", f"{self.relative_dtw[e][0]}",
    #                               f"{self.confidence[e]}", f"{self.original_label}",
    #                               f"{self.final_auc[e]}", f"{self.insertion[e]}", f"{self.deletion[e]}"]) + "\n")

    def to_csv(self, SAVE_DIR, TAG, UUID, SAMPLE_ID):
        save_path = f"{SAVE_DIR}/perturbations-{TAG}-{UUID}-{SAMPLE_ID}.csv"
        print(f"Saving perturbations to {save_path}")
        with open(save_path, "w") as f:
            f.write(",".join(self.column_names) + "\n")
            for e, row in enumerate(self.rows):
                f.write(",".join([str(i) for i in row]) + "\n")
