# # -*- coding: utf-8 -*-
# """
# | **@created on:** 9/22/20,
# | **@author:** prathyushsp,
# | **@version:** v0.0.1
# |
# | **Description:**
# |
# |
# | **Sphinx Documentation Status:**
# """

# __all__ = ['parse_arguments']

# import argparse
# from nte.experiment.utils import str2bool


# def parse_arguments(standalone=False):

#     parser = argparse.ArgumentParser(description='NTE Pipeline')

#     # General Configuration
#     parser.add_argument('--pname', type=str, help='Project name - [project_name]', default="CaseStudy_mc")
#     parser.add_argument('--task_id', type=int, help='Task ID', default=0)
#     parser.add_argument('--run_id', type=str, help='Run ID', default=19)
#     parser.add_argument('--save_perturbations', type=str2bool, nargs='?', const=False, help='Save Perturbations',
#                         default=False)
#     parser.add_argument('--conf_thres', type=float, help="Confidence threshold of prediction", default=0.0)

#     # Run Configuration
#     parser.add_argument('--run_mode', type=str, help='Run Mode - ["single", "local", "turing"]',default='single', choices=["single", "local", "turing"])
#     parser.add_argument('--dataset_type', type=str, help='Run Mode - ["train", "test", "valid"]',default='test', choices=["train", "test", "valid"])
#     parser.add_argument('--samples_per_task', type=int, help='Number of samples to run per task in turing mode',default=10)
#     parser.add_argument('--jobs_per_task', type=int, help='Max number of jobs to run all samples',default=-1)
#     # parser.add_argument('--single_sample_id', type=int, help='Single Sample',default=25) #24
#     # parser.add_argument('--single_sample_id', type=int, help='Single Sample',default=1) #24
#     parser.add_argument('--single_sample_id', type=int, help='Single Sample',default=34) #24 #50 # 180 #240 #34 ACSF  and 77

#     # Seed Configuration
#     parser.add_argument('--enable_seed', type=str2bool, nargs='?', const=True, help='Enable Seed',default=True)
#     parser.add_argument('--enable_seed_per_instance', type=str2bool, nargs='?', const=True,help='Enable Seed Per Instance',default=False)
#     parser.add_argument('--seed_value', type=int, help='Seed Value',default=0)

#     # Mask Normalization
#     parser.add_argument('--mask_norm', type=str, help='Mask Normalization - ["clamp", "softmax", "sigmoid"]',default='clamp',choices=["clamp", "softmax", "sigmoid", "none"])

#     # Algorithm
#     parser.add_argument('--algo', type=str, help='Algorithm type required - [mse|cm|nte|lime|shap|rise]',
#                         default='pert-mc',
#                         # default='distinct',
#                         # default='rise-rep',
#                         choices=["mse", "cm", "lime", "nte", "shap", "rise", "random", "rise-rep", "p-nte", "p-nte-v1",
#                                  "p-nte-v2", "p-nte-v3",  "p-nte-v4", "p-nte-v5", "cmwr", "nte-kl", "nte-dual", "pert", "pert-mv", "pert-mc","distinct"])
#     parser.add_argument('--grad_replacement', type=str, help='Gradient Based technique replacement strategy',
#                         # default='random_opposing_instance',
#                         default='random_instance',
#                         # default='min_class_random_instance',
#                         # default='max_class_random_instance',
#                         # default='min_class_mean',
#                         # default='next_class_random_instance',
#                         choices=["zeros", "class_mean", "instance_mean", "random_instance", "random_opposing_instance","min_class_random_instance","max_class_random_instance","min_class_mean","next_class_random_instance"])
#     parser.add_argument('--dynamic_replacement', type=str2bool, nargs='?', const=True,
#                         help='Dynamic pick of Zt on every epoch', default=True)
#     parser.add_argument('--r_index', type=int, help='Replacement Index',
#                         default=1)
#                         # default=139)

#     # Dataset and Background Dataset configuration
#     parser.add_argument('--dataset', type=str, default='Plane-mc',
#                         help='Dataset name required - [blip|wafer|gun_point|ford_a|ford_b|earthquakes|ptb|ecg]',
#                         choices=["blip", "wafer", "cricketx", "gun_point", "earthquakes", "computers",
#                                  "ford_a", "ford_b", "ptb","ecg", "cricketx-mc","AbnormalHeartbeat","ACSF","blip-mc","EOGHorizontalSignal-mc","Plane-mc","SmallKitchenAppliances-mc","Trace-mc","Rock-mc","ECG5000-mc","Meat-mc",
#                                  "1", "2", "3", "4", "5", "6", "7","8","9", "10","11","12","13","14","15","16","17","18","19","20"])
#     parser.add_argument('--background_data', type=str, help='[train|test|none]', default='test',
#                         choices=["train", "test", "none"])
#     parser.add_argument('--background_data_perc', type=float, help='%% of Background Dataset', default=100)

#     # Black-box model configuration
#     parser.add_argument('--bbmg', type=str, help='Grouped Time steps [yes|no]', default='no',choices=["yes", "no"])
#     parser.add_argument('--bbm', type=str, help='Black box model type - [dnn|rnn]', default='dnn',choices=["dnn", "rnn"])
#     parser.add_argument('--bbm_path', type=str, help='Black box model path - [dnn|rnn]', default="default")
#     parser.add_argument('--enable_cosine_loss', type=str, help='Enable cosine Loss', default='no', choices=["yes","no"])
#     parser.add_argument('--enable_hinge_loss', type=str, help='Enable Hinge Loss', default='no', choices=["yes","no"])
#     parser.add_argument('--enable_triplet_loss', type=str, help='Enable Triplet Loss', default='no', choices=["yes","no"])
#     # Gradient Based Algo configurations
#     parser.add_argument('--enable_blur', type=str2bool, nargs='?', const=True, help='Enable blur', default=False)
#     parser.add_argument('--enable_tvnorm', type=str2bool, nargs='?', const=True, help='Enable TV Norm', default=False)
#     parser.add_argument('--enable_budget', type=str2bool, nargs='?', const=True, help='Enable budget', default=False)
#     parser.add_argument('--mask_mse_coeff', type=float, help='MSE Coefficient', default=1.0) #0.20 is best
#     parser.add_argument('--enable_noise', type=str2bool, nargs='?', const=True, help='Enable Noise', default=True)
#     parser.add_argument('--enable_dist', type=str2bool, nargs='?', const=True, help='Enable Dist Loss', default=False)
#     # parser.add_argument('--enable_dtw', type=str2bool, nargs='?', const=True, help='Enable DTW', default=False)
#     # parser.add_argument('--enable_weighted_dtw', type=str2bool, nargs='?', const=True, help='Enable DTW', default=False)
#     # parser.add_argument('--enable_euclidean_loss', type=str2bool, nargs='?', const=True, help='Enable EUC Loss',
#     #                     default=False)
#     # parser.add_argument('--enable_weighted_euclidean_loss', type=str2bool, nargs='?', const=True, help='Enable W EUC Loss',
#     #                     default=False)
#     parser.add_argument('--enable_lr_decay', type=str2bool, nargs='?', const=True, help='LR Decay', default=False)

#     parser.add_argument('--dist_loss', type=str, help='Distance Loss Type - ["euc", "dtw", "w_euc", "w_dtw"]',
#                         default='euc',
#                         # default='dtw',
#                         choices=["euc", "dtw", "w_euc", "w_dtw", "n_dtw", "n_w_dtw", "no_dist"])

#     parser.add_argument('--early_stop_criteria_perc', type=float, help='Early Stop Criteria Percentage',default=0.80)

#     # Evaluation Metric Configuration
#     parser.add_argument('--run_eval', type=str2bool, nargs='?', const=True, help='Run Evaluation Metrics',
#                         default=True)
#     parser.add_argument('--run_eval_every_epoch', type=str2bool, nargs='?', const=True,
#                         help='Run Evaluation Metrics for every epoch',
#                         default=False)
#     parser.add_argument('--eval_replacement', type=str,
#                         help='Replacement Timeseries for evaluation [zeros|class_mean|instance_mean]',
#                         default='class_mean',
#                         choices=["zeros", "class_mean", "instance_mean"])
#     parser.add_argument('--eval_replacement_mc', type=str,
#                         help='Replacement Timeseries for MC evaluation [zeros|ones|class_mean|min_class_mean|instance_mean| random_class_random_index | min_class_random_index| max_class_random_index| max_class_max_index | random_class_mean | cluster_mean]',
#                         default='cluster_mean',
#                         # default='min_class_random_index',
#                         # default = 'max_class_max_index',
#                         # default = 'instance_mean',
#                         choices=["zeros", "ones", "class_mean", "min_class_mean", "instance_mean", "random_class_random_index", "min_class_random_index", "max_class_random_index", "max_class_max_index", "random_class_mean", "cluster_mean"])

#     # Hyper Param Configuration
#     parser.add_argument('--lr', type=float, help='Learning Rate', default=0.001) #0.01
#     parser.add_argument('--lr_decay', type=float, help='LR Decay', default=0.999)
#     parser.add_argument('--l1_coeff', type=float, help='L1 Coefficient', default=1) #0.05
#     parser.add_argument('--tv_coeff', type=float, help='TV Norm Coefficient', default=0.1)
#     parser.add_argument('--tv_beta', type=float, help='TV Norm Beta', default=3)
#     parser.add_argument('--dist_coeff', type=float, help='Dist Loss Coeff', default=1)
#     parser.add_argument('--w_decay', type=float, help='Weight Decay', default=0.0)
#     # parser.add_argument('--dtw_coeff', type=float, help='Soft DTW Coeff', default=1)
#     # parser.add_argument('--euc_coeff', type=float, help='EUC Loss Coeff', default=1)
#     parser.add_argument('--mse_coeff', type=float, help='MSE Coefficient', default=1.0) #0.20 is best
#     #185
#     parser.add_argument('--bwin', type=int, help='bwin ', default=1)
#     parser.add_argument('--run', type=str, help='Run ', default=501 )
#     parser.add_argument('--bsigma', type=float, help='bsigma ', default=0.9)
#     parser.add_argument('--sample', type=float, help='sample', default=6.25)
#     parser.add_argument('--multi_class', type=str,help='multi class',default='yes', choices=["yes", "no"])
#     parser.add_argument('--ce', type=str,help='CrossEntropy', default='no', choices=["yes", "no"])
#     parser.add_argument('--mse', type=str,help='MSE ', default='yes', choices=["yes", "no"])
#     parser.add_argument('--kl', type=str,help='KLDiv', default='yes', choices=["yes", "no"])
#     parser.add_argument('--ml', type=str,help='MultiLabel', default='no', choices=["yes", "no"])
#     parser.add_argument('--first_class', type=str,help='First class ',default='no', choices=["yes", "no"])
#     parser.add_argument('--second_class', type=str,help='Second class ',default='no', choices=["yes", "no"])
#     parser.add_argument('--third_class', type=str,help='Third class ',default='no', choices=["yes", "no"])
#     parser.add_argument('--top_n', type=str,help='Top N classes ',default='yes', choices=["yes", "no"])
#     parser.add_argument('--print', type=str,help='Print',default='yes', choices=["yes", "no"])
#     parser.add_argument('--ssd', type=str,help='shared saliency deletion ',default='dynamic', choices=["manual", "dynamic","distinct","prob_distinct"])
#     parser.add_argument('--top_class', type=str,help='only top class mux or not',default='yes', choices=["yes", "no"])
#     parser.add_argument('--old_auc', type=str,help='Old AUC or MC AUC ',default='no', choices=["yes", "no"])
#     parser.add_argument('--new_auc', type=str,help='New AUC first mask for MC AUC ',default='yes', choices=["yes", "no"])
#     parser.add_argument('--preserve_prob', type=str,help='Preserve original probability ',default='no', choices=["yes", "no"])
#     parser.add_argument('--tsne', type=str,help='TSNE plots',default='no', choices=["yes", "no"])
#     # Early Stopping Criteria
#     parser.add_argument('--early_stopping', type=str2bool, nargs='?', const=False, help='Enable or Disable Early Stop',default=False)
#     parser.add_argument('--early_stop_min_epochs', type=float, help='Early Stop Minimum Epochs',default=000)
#     parser.add_argument('--max_itr', type=int, help='Maximum Iterations', default=2000)
#     # parser.add_argument('--early_stop_prob_range', type=float, help='Early Stop probability range',default=0.30)
#     parser.add_argument('--early_stop_prob_range', type=float, help='Early Stop probability range',default=0.10)
#     parser.add_argument('--early_stop_diff', type=float, help='Early Stop Difference',default=00)
#     parser.add_argument('--prob_upto', type=float,help='Probability upto',default=0.00)
#     # parser.add_argument('--prob_upto', type=float,help='Probability1upto',default=1.00)
#     parser.add_argument('--num_perturbs', type=float,help='Number of Perturbations',default=500)
#     parser.add_argument('--sample_masks', type=int,help='Number of Sample masks ',default=1)
#     parser.add_argument('--n_clusters', type=int,help='Number of clusters ',default=1)
#     parser.add_argument('--cluster_indx', type=int,help='Which cluster index to use ',default=0)
#     parser.add_argument('--clusters', type=str,help='yes or mean ',default='yes', choices=["yes", "mean"])
#     parser.add_argument('--cluster_dtw', type=str,help='softdtw or dtw ',default='dtw', choices=["softdtw", "dtw"])
#     parser.add_argument('--window_size', type=int,help='Which window size to use ',default=1)
#     parser.add_argument('--nth_prob', type=float,help='nth_prob',default=0.1)
#     parser.add_argument('--class_prob', type=float,help='class_prob',default=0.0)
#     parser.add_argument('--p_buf_init', type=str, help='random | maha | euc | cluster',
#                         default='cluster', choices=["random", "maha", "euc", "cluster"])
#     parser.add_argument('--prob_loss', type=str, help='yes | no',
#                         default='no', choices=["yes", "no"])
#     parser.add_argument('--num_masks', type=int,help='Number of Random Masks',default=1000)

#     # parser.add_argument('--timesteps', type=int,help='Number of time steps ',default=275)

#     if standalone:
#         return parser.parse_known_args()
#     else:
#         return parser.parse_args()


# -*- coding: utf-8 -*-
"""
| **@created on:** 9/22/20,
| **@author:** prathyushsp,
| **@version:** v0.0.1
|
| **Description:**
|
|
| **Sphinx Documentation Status:**
"""

__all__ = ['parse_arguments']

import argparse
from nte.experiment.utils import str2bool


def parse_arguments(standalone=False):

    parser = argparse.ArgumentParser(description='NTE Pipeline')

    # General Configuration
    parser.add_argument('--pname', type=str, help='Project name - [project_name]', default="tsmule_mc")
    parser.add_argument('--task_id', type=int, help='Task ID', default=0)
    parser.add_argument('--run_id', type=str, help='Run ID', default=19)
    parser.add_argument('--save_perturbations', type=str2bool, nargs='?', const=False, help='Save Perturbations',
                        default=False)
    parser.add_argument('--conf_thres', type=float, help="Confidence threshold of prediction", default=0.0)

    # Run Configuration
    parser.add_argument('--run_mode', type=str, help='Run Mode - ["single", "local", "turing"]',default='single', choices=["single", "local", "turing"])
    parser.add_argument('--dataset_type', type=str, help='Run Mode - ["train", "test", "valid"]',default='test', choices=["train", "test", "valid"])
    parser.add_argument('--samples_per_task', type=int, help='Number of samples to run per task in turing mode',default=1)
    parser.add_argument('--jobs_per_task', type=int, help='Max number of jobs to run all samples',default=1)
    #KDD Case study do not change Meat-mc default sample  = 1, 37 and  39 for red or 19 for LMUX CASE study
    parser.add_argument('--single_sample_id', type=int, help='Single Sample',default=19)

    # Seed Configuration
    parser.add_argument('--enable_seed', type=str2bool, nargs='?', const=True, help='Enable Seed',default=True)
    parser.add_argument('--enable_seed_per_instance', type=str2bool, nargs='?', const=True,help='Enable Seed Per Instance',default=False)
    parser.add_argument('--seed_value', type=int, help='Seed Value',default=1)

    # Mask Normalization
    parser.add_argument('--mask_norm', type=str, help='Mask Normalization - ["clamp", "softmax", "sigmoid"]',default='clamp',choices=["clamp", "softmax", "sigmoid", "none"])

    # Algorithm
    parser.add_argument('--algo', type=str, help='Algorithm type required - [mse|cm|nte|lime|shap|rise|lmux|lmux_v1]',
                        #default='lmux_v1',
                        default='tsmule',
                        choices=["mse", "cm", "lime", "shap", "rise", "random", "rise-rep", "pert",
                        "pert-mc","distinct", "lmux","dyna", "lmux_v1", "tsmule", "timeshap"])
    parser.add_argument('--grad_replacement', type=str, help='Gradient Based technique replacement strategy',
                        default='random_instance',
                        # default='zeros',
                        choices=["zeros", "class_mean", "instance_mean", "random_instance", "random_opposing_instance","min_class_random_instance","max_class_random_instance","min_class_mean","next_class_random_instance"])
    parser.add_argument('--dynamic_replacement', type=str2bool, nargs='?', const=True,
                        help='Dynamic pick of Zt on every epoch', default=True)
    parser.add_argument('--r_index', type=int, help='Replacement Index',
                        default=1)

    # Dataset and Background Dataset configuration
    parser.add_argument('--dataset', type=str, default='Meat-mc',
                        help='Dataset name required - [blip|wafer|gun_point|ford_a|ford_b|earthquakes|ptb|ecg]',
                        choices=["blip", "wafer", "cricketx", "gun_point", "earthquakes", "computers",
                                 "ford_a", "ford_b", "ptb","ecg", "cricketx-mc","AbnormalHeartbeat","ACSF","blip-mc","EOGHorizontalSignal-mc","Plane-mc","SmallKitchenAppliances-mc","Trace-mc","Rock-mc","ECG5000-mc","Meat-mc", "MixedShapes-mc",
                                 "1", "2", "3", "4", "5", "6", "7","8","9", "10","11","12","13","14","15","16","17","18","19","20", "21"])
    parser.add_argument('--background_data', type=str, help='[train|test|none]', default='test',
                        choices=["train", "test", "none"])
    parser.add_argument('--background_data_perc', type=float, help='%% of Background Dataset', default=100)

    # Black-box model configuration
    # parser.add_argument('--bbmg', type=str, help='Grouped Time steps [yes|no]', default='no',choices=["yes", "no"])
    parser.add_argument('--bbm', type=str, help='Black box model type - [dnn|rnn|cnn]', default='dnn',choices=["dnn", "rnn", "cnn"])
    parser.add_argument('--bbm_path', type=str, help='Black box model path - [dnn|rnn]', default="default")

    # Gradient Based Algo configurations
    parser.add_argument('--enable_blur', type=str2bool, nargs='?', const=True, help='Enable blur', default=False)
    parser.add_argument('--enable_tvnorm', type=str2bool, nargs='?', const=True, help='Enable TV Norm', default=True)
    parser.add_argument('--enable_budget', type=str2bool, nargs='?', const=True, help='Enable budget', default=True)
    parser.add_argument('--mask_mse_coeff', type=float, help='MSE Coefficient', default=1.0) #0.20 is best
    parser.add_argument('--enable_noise', type=str2bool, nargs='?', const=True, help='Enable Noise', default=False)
    parser.add_argument('--enable_dist', type=str2bool, nargs='?', const=True, help='Enable Dist Loss', default=False)

    parser.add_argument('--enable_lr_decay', type=str2bool, nargs='?', const=True, help='LR Decay', default=False)

    parser.add_argument('--dist_loss', type=str, help='Distance Loss Type - ["euc", "dtw", "w_euc", "w_dtw"]',
                        default='no_dist',
                        # default='dtw',
                        choices=["euc", "dtw", "w_euc", "w_dtw", "n_dtw", "n_w_dtw", "no_dist"])

    parser.add_argument('--early_stop_criteria_perc', type=float, help='Early Stop Criteria Percentage',default=0.80)

    # Evaluation Metric Configuration
    parser.add_argument('--run_eval', type=str2bool, nargs='?', const=True, help='Run Evaluation Metrics',
                        default=True)
    parser.add_argument('--run_eval_every_epoch', type=str2bool, nargs='?', const=True,
                        help='Run Evaluation Metrics for every epoch',
                        default=False)
    parser.add_argument('--eval_replacement', type=str,
                        help='Replacement Timeseries for evaluation [zeros|class_mean|instance_mean]',
                        default='class_mean',
                        choices=["zeros", "class_mean", "instance_mean"])
    parser.add_argument('--eval_replacement_mc', type=str,
                        help='Replacement Timeseries for MC evaluation [zeros|ones|class_mean|min_class_mean|instance_mean| random_class_random_index | min_class_random_index| max_class_random_index| max_class_max_index | random_class_mean | cluster_mean]',
                        # default='min_class_mean',
                        default='cluster_mean',
                        choices=["zeros", "ones", "class_mean", "min_class_mean", "instance_mean", "random_class_random_index", "min_class_random_index", "max_class_random_index", "max_class_max_index", "random_class_mean", "cluster_mean"])

    # Hyper Param Configuration
    parser.add_argument('--lr', type=float, help='Learning Rate', default=0.001) #0.01
    parser.add_argument('--lr_decay', type=float, help='LR Decay', default=0.999)

    parser.add_argument('--l_prev', type=str,
                        help='Type of L Preserve Loss [kld|mse]',default='kld',choices=["kld", "mse"])
    parser.add_argument('--l_prev_coeff', type=float, help='L Preservation Coefficient', default=1.0)  # 0.20 is best
    parser.add_argument('--l_budget_coeff', type=float, help='L Budget Coefficient', default=0.6)  # 0.05
    parser.add_argument('--l_tv_norm_coeff', type=float, help='L TV Norm Coefficient', default=0.5)
    parser.add_argument('--tv_beta', type=float, help='TV Norm Beta', default=3)
    parser.add_argument('--l_ssd_coeff', type=float, help='L SSD Coefficient', default=0.01)
    parser.add_argument('--l_min_coeff', type=float, help='L Minimize Coefficient', default=1.0)
    parser.add_argument('--l_max_coeff', type=float, help='L Minimize Coefficient', default=0.7)
    parser.add_argument('--noise_mean', type=float, help='Noise Mean', default=0)
    parser.add_argument('--noise_std', type=float, help='Noise Std', default=0.1)

    # mask repo config
    parser.add_argument('--enable_mask_repo', type=str2bool, nargs='?', const=True, help='Enable Mask Repo', default=True)
    parser.add_argument('--mask_repo_type', type=str, help='mask repo type', default="last_n_cond",
                        choices=["last_n_cond", "last_n", "min_l_total", "min_l_prev", "max_conf"])
    parser.add_argument('--mask_repo_rec', type=int, help='Fetch N masks', default=10)


    parser.add_argument('--dist_coeff', type=float, help='Dist Loss Coeff', default=0)
    parser.add_argument('--w_decay', type=float, help='Weight Decay', default=0.0)

    parser.add_argument('--bwin', type=int, help='bwin ', default=1)
    parser.add_argument('--run', type=str, help='Run ', default=501 )
    parser.add_argument('--bsigma', type=float, help='bsigma ', default=0.9)
    parser.add_argument('--sample', type=float, help='sample', default=6.25)
    parser.add_argument('--multi_class', type=str,help='multi class',default='yes', choices=["yes", "no"])

    parser.add_argument('--ssd', type=str,help='shared saliency deletion ',default='dynamic', choices=["manual", "dynamic","distinct","prob_distinct"])

    parser.add_argument('--tsne', type=str,help='TSNE plots',default='no', choices=["yes", "no"])
    # Early Stopping Criteria
    parser.add_argument('--early_stopping', type=str2bool, nargs='?', const=False, help='Enable or Disable Early Stop',default=False)
    parser.add_argument('--early_stop_min_epochs', type=float, help='Early Stop Minimum Epochs',default=0)
    parser.add_argument('--max_itr', type=int, help='Maximum Iterations', default=5000)
    parser.add_argument('--early_stop_prob_range', type=float, help='Early Stop probability range',default=0.10)
    parser.add_argument('--early_stop_diff', type=float, help='Early Stop Difference',default=0)
    parser.add_argument('--prob_upto', type=float,help='Probability upto',default=0.00)
    parser.add_argument('--sample_masks', type=int,help='Number of Sample masks ',default=1)
    parser.add_argument('--window_size', type=int,help='Which window size to use ',default=1)
    parser.add_argument('--class_prob', type=float,help='class_prob',default=0.0)
    parser.add_argument('--p_buf_init', type=str, help='random | maha | euc | cluster',
                        default='cluster', choices=["random", "maha", "euc", "cluster"])
    parser.add_argument('--num_masks', type=int,help='Number of Random Masks',default=1000)

    # Extra arguments - Will be deprecated
    parser.add_argument('--n_clusters', type=int,help='Number of clusters ',default=1)
    parser.add_argument('--cluster_indx', type=int,help='Which cluster index to use ',default=0)
    parser.add_argument('--clusters', type=str,help='yes or mean ',default='yes', choices=["yes", "mean"])
    parser.add_argument('--cluster_dtw', type=str,help='softdtw or dtw ',default='dtw', choices=["softdtw", "dtw"])
    parser.add_argument('--prob_loss', type=str, help='yes | no', default='no', choices=["yes", "no"])
    parser.add_argument('--nth_prob', type=float,help='nth_prob',default=0.1)
    parser.add_argument('--timesteps', type=int,help='Number of time steps ',default=275)
    parser.add_argument('--num_perturbs', type=float,help='Number of Perturbations',default=500)
    parser.add_argument('--top_class', type=str,help='only top class mux or not',default='yes', choices=["yes", "no"])
    parser.add_argument('--old_auc', type=str,help='Old AUC or MC AUC ',default='no', choices=["yes", "no"])
    parser.add_argument('--new_auc', type=str,help='New AUC first mask for MC AUC ',default='yes', choices=["yes", "no"])
    parser.add_argument('--preserve_prob', type=str,help='Preserve original probability ',default='no', choices=["yes", "no"])
    parser.add_argument('--ce', type=str,help='CrossEntropy', default='no', choices=["yes", "no"])
    parser.add_argument('--mse', type=str,help='MSE ', default='yes', choices=["yes", "no"])
    parser.add_argument('--kl', type=str,help='KLDiv', default='yes', choices=["yes", "no"])
    parser.add_argument('--ml', type=str,help='MultiLabel', default='no', choices=["yes", "no"])
    parser.add_argument('--first_class', type=str,help='First class ',default='no', choices=["yes", "no"])
    parser.add_argument('--second_class', type=str,help='Second class ',default='no', choices=["yes", "no"])
    parser.add_argument('--third_class', type=str,help='Third class ',default='no', choices=["yes", "no"])
    parser.add_argument('--top_n', type=str,help='Top N classes ',default='yes', choices=["yes", "no"])
    parser.add_argument('--print', type=str,help='Print',default='yes', choices=["yes", "no"])
    parser.add_argument('--dtw_coeff', type=float, help='Soft DTW Coeff', default=1)
    parser.add_argument('--euc_coeff', type=float, help='EUC Loss Coeff', default=1)
    parser.add_argument('--enable_dtw', type=str2bool, nargs='?', const=True, help='Enable DTW', default=False)
    parser.add_argument('--enable_weighted_dtw', type=str2bool, nargs='?', const=True, help='Enable DTW', default=False)
    parser.add_argument('--enable_euclidean_loss', type=str2bool, nargs='?', const=True, help='Enable EUC Loss',
                        default=False)
    parser.add_argument('--enable_weighted_euclidean_loss', type=str2bool, nargs='?', const=True, help='Enable W EUC Loss',
                        default=False)
    parser.add_argument('--enable_cosine_loss', type=str, help='Enable cosine Loss', default='no', choices=["yes","no"])
    parser.add_argument('--enable_hinge_loss', type=str, help='Enable Hinge Loss', default='no', choices=["yes","no"])
    parser.add_argument('--enable_triplet_loss', type=str, help='Enable Triplet Loss', default='no', choices=["yes","no"])
    parser.add_argument('--mse_coeff', type=float, help='MSE Coefficient', default=1.0)  # 0.20 is best
    parser.add_argument('--bbmg', type=str, help='Grouped Time steps [yes|no]', default='no', choices=["yes", "no"])
    parser.add_argument('--l1_coeff', type=float, help='L1 Coefficient', default=1)  # 0.05
    parser.add_argument('--tv_coeff', type=float, help='TV Norm Coefficient', default=0.1)
    parser.add_argument('--iou_alpha', type=float, help='IOU alpha ', default=0.5)

    if standalone:
        return parser.parse_known_args()
    else:
        return parser.parse_args()
