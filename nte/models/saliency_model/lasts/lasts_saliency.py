import os,sys,inspect
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
# parentdir = os.path.dirname(currentdir)
# sys.path.insert(0,parentdir)
# from datasets import build_cbf
from joblib import load
# from blackbox_wrapper import BlackboxWrapper
from lasts import Lasts
import numpy as np
import keras
from nte.models.saliency_model.lasts.LASTS_explainer.utils import reconstruction_accuracy_vae, choose_z
from nte.models.saliency_model.lasts.LASTS_explainer.variational_autoencoder import load_model
from nte.models.saliency_model.lasts.LASTS_explainer.neighborhood_generators import NeighborhoodGenerator
from nte.models.saliency_model.lasts.LASTS_explainer.utils import plot_labeled_latent_space_matrix
import seaborn as sns
from nte.models.saliency_model.lasts.LASTS_explainer.saxdt import Saxdt
from nte.models.saliency_model.lasts.LASTS_explainer.sbgdt import Sbgdt

from nte.models.saliency_model import Saliency
import numpy as np
import torch
from nte.utils.perturbation_manager import PerturbationManager
from nte.experiment.evaluation import run_evaluation_metrics

class LastsSaliency(Saliency):

    def __init__(self, background_data, background_label, predict_fn, args, max_itr=1):
        super().__init__(background_data=background_data, background_label=background_label, predict_fn=predict_fn)
        self.confidences = []
        self.perturbations = []
        self.max_itr = max_itr
        self.args = args
        self.softmax_fn = torch.nn.Softmax(dim=-1)

    def generate_saliency(self, data, label, **kwargs):

        # IMPORT DATASET
        # random_state = 0
        # dataset_name = "cbf"

        # (X_train, y_train, X_val, y_val,
        # X_test, y_test, X_exp_train, y_exp_train,
        # X_exp_val, y_exp_val, X_exp_test, y_exp_test) = build_cbf(n_samples=600, random_state=random_state)

        # IMPORT BLACKBOX
        # knn = load(parentdir + "/trained_models/cbf/cbf_knn.joblib")
        # resnet = keras.models.load_model(parentdir + "/trained_models/cbf/cbf_resnet.h5")

        # blackbox = knn

        # WRAP BLACKBOX
        # blackbox = BlackboxWrapper(knn, 2, 1)

        # IMPORT AUTOENCODER
        if self.args.dataset == "GunPoint":
            # Load gunpoint VAE
            _, _, autoencoder = load_model(parentdir + "/trained_models/cbf/cbf_vae")

        encoder = autoencoder.layers[2]
        decoder = autoencoder.layers[3]

        # CHECK RECONSTRUCTION ACCURACY
        print(reconstruction_accuracy_vae(X_exp_test, encoder, decoder, blackbox))

        # CHOOSE INSTANCE TO EXPLAIN
        i = 0
        x = X_exp_test[i].ravel().reshape(1, -1, 1)
        z = choose_z(x, encoder, decoder, n=1000, x_label=blackbox.predict(x)[0], blackbox=blackbox, check_label=True)
        z_label = blackbox.predict(decoder.predict(z))[0]

        K = encoder.predict(X_exp_train)

        neighborhood_generator = NeighborhoodGenerator(blackbox, decoder)
        neigh_kwargs = {
            "balance": False,
            "n": 500,
            "n_search": 10000,
            "threshold": 2,
            "sampling_kind": "uniform_sphere",
            "kind": "gaussian_matched",
            "verbose": True,
            "stopping_ratio": 0.01,
            "downward_only": True,
            "redo_search": True,
            "forced_balance_ratio": 0.5,
            "cut_radius": True
        }

        lasts_ = Lasts(blackbox,
                    encoder,
                    decoder,
                    x,
                    neighborhood_generator,
                    z_fixed=z,
                    labels=["cylinder", "bell", "funnel"]
                    )

        # GENERATE NEIGHBORHOOD
        # out = lasts_.generate_neighborhood(**neigh_kwargs)

        # VARIOUS PLOTS
        # lasts_.plot_exemplars_and_counterexemplars()
        # lasts_.plot_latent_space(scatter_matrix=True)
        # lasts_.plot_latent_space(scatter_matrix=True, K=K)
        # lasts_.morphing_matrix()

        surrogate = Sbgdt(shapelet_model_params={"max_iter": 50}, random_state=0)
        # surrogate = Saxdt(random_state=np.random.seed(0))
        # WARNING: you need a forked version of the library sktime in order to view SAX plots

        # SUBSEQUENCE EXPLAINER
        lasts_.fit_surrogate(surrogate, binarize_labels=True)

        # SUBSEQUENCE TREE
        # lasts_.surrogate._graph

        # VARIOUS PLOTS
        # lasts_.plot_binary_heatmap(step=5)
        # lasts_.plot_factual_and_counterfactual()




        category = np.argmax(kwargs['target'].cpu().data.numpy())
        self.perturbation_manager = PerturbationManager(
            original_signal=data.flatten(),
            algo="random", prediction_prob=np.max(kwargs['target'].cpu().data.numpy()), original_label=label,
            sample_id=self.args.single_sample_id)
        for i in range(self.max_itr):
            saliency = np.random.random(len(data.flatten()))
            perturbated_input = torch.tensor(data * saliency, dtype=torch.float32)
            confidence = float(self.softmax_fn(self.predict_fn(perturbated_input))[0][category].item())
            print(f"Generating random saliency  - Itr: {i} | Confidence: {confidence:.4f}")

            if kwargs['save_perturbations'] and not self.args.run_eval_every_epoch:
                self.perturbation_manager.add_perturbation(
                    perturbation=perturbated_input.cpu().detach().numpy().flatten(),
                    step=i, confidence=confidence, saliency=saliency)

            if self.args.run_eval_every_epoch:
                metrics = {
                    'eval_metrics': run_evaluation_metrics(self.args.eval_replacement, kwargs['dataset'],
                                                           data.flatten(),
                                                           self.predict_fn, saliency,
                                                           kwargs['save_dir'], False)}

                if kwargs['save_perturbations']:
                    self.perturbation_manager.add_perturbation(perturbation=perturbated_input.flatten(),
                                                               step=i, confidence=confidence, saliency=saliency,
                                                               insertion=metrics['eval_metrics']['Insertion']['trap'],
                                                               deletion=metrics['eval_metrics']['Deletion']['trap'],
                                                               final_auc=metrics['eval_metrics']['Final']['AUC'],
                                                               saliency_sum=float(np.sum(saliency)),
                                                               saliency_var=float(np.var(saliency))
                                                               )
        return saliency, self.perturbation_manager
