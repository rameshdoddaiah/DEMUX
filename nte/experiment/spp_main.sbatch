#!/bin/bash
#SBATCH -p short
#SBATCH -N 1                      # number of nodes
#SBATCH -n 4                      # number of cores
#SBATCH --mem=8GB               # memory pool for all cores
#SBATCH -t 0-24:00                # time (D-HH:MM)
#SBATCH --checkpoint=5
#SBATCH --checkpoint-dir=checkpoints
#SBATCH --gres=gpu:0              # number of GPU
#SBATCH --job-name=main
#SBATCH -o logs/slurm-main-output_%A-%a    # STDOUT
#SBATCH -e logs/slurm-main-error_%A-%a     # STDERR
#SBATCH --mail-type=ALL
#SBATCH --mail-user=psparvatharaju@wpi.edu
##SBATCH --exclude=compute-0-27 
##SBATCH -C T4

## Dataset Configuration
## Wafer (152) | spt 2 | j 76
## Ford B (810) | spt 9 | j 90
## Ford A (1320) | spt 15 | j 88
## Earthquake (139) | spt 2 | j 70
## GunPoint (150) | spt 2 | j 75
## CricketX (61)
## Computers (250)
 
## ACSF (100)
## Plane-mc (105)
## Trace-mc (100)
## Meat-mc (60)
## ECG5000-mc (500)
## MixedShapes (2426)

#Order
# number_to_dataset = {
#     "1": "wafer",
#     "2": "cricket_x",
#     "3": "gun_point",
#     "4": "earthquakes",
#     "5": "computers",
#     "6": "ford_a",
#     "7": "ford_b"
# }

python3 main.py --pname lmux_v1_auc_table --task_id ${SLURM_ARRAY_TASK_ID}\
                --run_mode turing\
                --jobs_per_task -1 \
                --samples_per_task 2\
                --dataset Plane-mc\
                --algo lmux_v1\
                --seed_value 2\
                --run 2\
                --bbm rnn\
                --print no\
                --l_budget_coeff 0.05\
                --l_tv_norm_coeff 0.1\
                --l_ssd_coeff 0.01\

#                --samples_per_task 2 \
#                --enable_mask_repo true\
#                --mask_repo_type min_l_total\
#                --mask_repo_rec 5\
#                --max_itr 5000\
#                --bbm dnn\
#                --l_max_coeff 1\
#                --l_ssd_coeff 0.5\
#                --l_prev_coeff 1.0\
#                --l_budget_coeff 5.0\
#                --l_tv_norm_coeff 1.0\
#                --noise_std 0.5\
#                --enable_tvnorm True\
#                --enable_budget True\
#                --enable_noise False\
#                --w_decay 0.00 \


#python3 main.py --pname lmux_v1 --task_id ${SLURM_ARRAY_TASK_ID} \
                --run_mode turing --samples_per_task 2 \
                 --dataset ACSF\
                 --algo lmux\
                 --run working-rep-csv\
                 --class_prob 1.0\
                 --seed_value 1\
                 --enable_dist False --dist_loss no_dist --dist_coeff 0 \
                 --enable_lr_decay False \
                 --grad_replacement random_instance \
                 --eval_replacement class_mean \
                 --background_data test --background_data_perc 100\
                 --run_eval True  --enable_seed True \
                 --max_itr 5000\
                 --ssd dynamic\
                 --run_id 1 \
                 --w_decay 0.00 \
                 --bbm dnn \
                 --eval_replacement_mc cluster_mean\
                 --multi_class yes\
                 --mse yes\
                 --first_class no\
                 --second_class no\
                 --third_class no\
                 --ce no\
                 --ml no\
                 --kl yes\
                 --num_perturbs 500\
                 --prob_upto 0.0 \
                 --early_stop_prob_range 0.05 \
                 --early_stop_diff 0\
                 --early_stop_min_epochs 0\
                 --early_stopping False\
                 --tsne no \
                 --mask_mse_coeff 0.1\
                 --old_auc no\
                 --new_auc yes\
                 --top_class yes\
                 --enable_hinge_loss no\
                 --enable_triplet_loss no\
                 --preserve_prob no\
                 --mse_coeff 1.0\
                 --enable_tvnorm True\
                 --enable_budget True\
                 --print yes\
                 --top_n yes\
                 --dataset_type test\
                 --sample_masks 1\
                 --clusters mean\
                 --n_clusters 1\
                 --cluster_indx 0\
                 --cluster_dtw dtw\
                 --bbmg no\
                 --window_size 1\
                 --nth_prob 0.0\
                 --p_buf_init cluster\
                 --prob_loss no\


# Multiclass experiments
#python3 main.py --pname Loss_Study --task_id ${SLURM_ARRAY_TASK_ID} \
                --run_mode turing --samples_per_task 2 \
                 --dataset ACSF\
                 --algo pert-mc\
                 --run kl_cs_0.1\
                 --class_prob 1.0\
                 --seed_value 1\
                 --enable_dist False --dist_loss no_dist --dist_coeff 0 \
                 --enable_lr_decay False \
                 --grad_replacement random_instance \
                 --eval_replacement class_mean \
                 --background_data test --background_data_perc 100\
                 --run_eval True  --enable_seed True \
                 --max_itr 5000\
                 --ssd dynamic\
                 --run_id 1 \
                 --w_decay 0.00 \
                 --bbm dnn \
                 --eval_replacement_mc min_class_mean\
                 --multi_class yes\
                 --mse yes\
                 --first_class no\
                 --second_class no\
                 --third_class no\
                 --ce no\
                 --ml no\
                 --kl yes\
                 --num_perturbs 500\
                 --prob_upto 0.0 \
                 --early_stop_prob_range 0.05 \
                 --early_stop_diff 0\
                 --early_stop_min_epochs 0\
                 --early_stopping False\
                 --tsne no \
                 --mask_mse_coeff 0.1\
                 --old_auc no\
                 --new_auc yes\
                 --top_class yes\
                 --enable_hinge_loss no\
                 --enable_triplet_loss no\
                 --preserve_prob no\
                 --mse_coeff 1.0\
                 --enable_tvnorm False\
                 --enable_budget False\
                 --print yes\
                 --top_n yes\
                 --dataset_type test\
                 --sample_masks 1\
                 --clusters mean\
                 --n_clusters 1\
                 --cluster_indx 0\
                 --cluster_dtw dtw\
                 --bbmg no\
                 --window_size 1\
                 --nth_prob 0.0\
                 --p_buf_init cluster\
                 --prob_loss no\



#119,141,45,105,104,57,68,121,35,32
# PBuf Experiments
#python3 main.py --pname abl_study --task_id ${SLURM_ARRAY_TASK_ID} \
                 --run_mode turing --jobs_per_task 40 \
                 --algo nte-dual --dataset 1 \
                 --enable_dist False --dist_loss euc --dist_coeff 0.005 \
                 --enable_lr_decay False \
                 --grad_replacement random_instance \
                 --eval_replacement class_mean \
                 --background_data test --background_data_perc 100 \
                 --run_eval True  --enable_seed False \
                 --run_id back \
                 --run 7\
                 --lr 0.001 \
                 --w_decay 0.0 \
                 --mse_coeff 2.0\
                 --l1_coeff 1.0\
                 --max_itr 5000 \


# Dist Exps
# range - 0.01 0.1 1 10
#python3 main.py --pname dist_exps --task_id ${SLURM_ARRAY_TASK_ID} \
                 --run_mode turing --samples_per_task 14 \
                 --algo nte --dataset earthquakes --bbm dnn --bbm_path earthquakes_dnn_ce.ckpt \
                 --enable_dist False --dist_loss no_dist --dist_coeff 0 \
                 --grad_replacement random_instance \
                 --eval_replacement class_mean \
                 --background_data test --background_data_perc 100 \
                 --run_eval True \
                 --run_id 1 \

# BaseLine Experiments
#python3 main.py --pname baselines --task_id ${SLURM_ARRAY_TASK_ID} \
                 --run_mode turing --samples_per_task 54 \
                 --algo rise-rep --dataset ford_b --bbm dnn --bbm_path ford_b_dnn_ce.ckpt \
                 --grad_replacement random_instance \
                 --eval_replacement class_mean \
                 --background_data test --background_data_perc 100 \
                 --run_eval True \
                 --run_id 1 \


# Paper Experiments
#python3 main.py --pname Paper --task_id ${SLURM_ARRAY_TASK_ID} \
                 --run_mode turing --samples_per_task 30 \
                 --algo cm --dataset wafer --bbm dnn --bbm_path wafer_dnn_ce.ckpt \
                 --grad_replacement random_instance \
                 --eval_replacement class_mean \
                 --background_data test --background_data_perc 100 \
                 --run_eval True \
                 --enable_seed False \
                 --seed_value 5 \
                 --run_id 5 \
                 --run 5 \


# Perturbation Experiment
#python3 main.py --pname perturbation_analysis_v1 --task_id ${SLURM_ARRAY_TASK_ID} \
                 --run_mode single --single_sample_id 121 \
                 --enable_seed True \
                 --algo nte --dataset wafer --bbm dnn --bbm_path wafer_dnn_ce.ckpt \
                 --grad_replacement random_instance \
                 --eval_replacement class_mean \
                 --background_data test --background_data_perc 100 \
                 --max_itr 50 \
                 --save_perturbations True \
                 --run_eval True \
                 --run_eval_every_epoch True \


# DTW Experiment - EarthQuakes Dataset
# 70 Jobs | Zeros | Zeros + DTW | Random | Random + DTW
#python3 main.py --pname dtw_experiments --task_id ${SLURM_ARRAY_TASK_ID} \
                 --run_mode turing --samples_per_task 2 \
                 --algo mse --dataset earthquakes --bbm dnn --bbm_path earthquakes_dnn_ce.ckpt \
                 --grad_replacement zeros \
                 --eval_replacement class \
                 --enable_dtw False --dtw_coeff 0.05 \
                 --background_data test --background_data_perc 100 \
                 --run_eval True

# DTW Experiment - Ford A
# 88 Jobs
# python3 main.py --pname dtw_experiments --task_id ${SLURM_ARRAY_TASK_ID} \
                 --run_mode turing --samples_per_task 15 \
                 --algo mse --dataset ford_a --bbm dnn --bbm_path ford_a_dnn_ce.ckpt \
                 --grad_replacement random \
                 --eval_replacement class \
                 --enable_dtw True --dtw_coeff 0.05 \
                 --background_data test --background_data_perc 100 \
                 --run_eval True

# DTW Experiment - Ford B
# 81 Jobs
#python3 main.py --pname ford_b_threshold --task_id ${SLURM_ARRAY_TASK_ID} \
                 --run_mode turing --samples_per_task 10 \
                 --algo mse --dataset ford_b --bbm dnn --bbm_path ford_b_dnn_ce.ckpt \
                 --grad_replacement random \
                 --eval_replacement class \
                 --enable_dtw True --dtw_coeff 0.05 \
                 --background_data test --background_data_perc 100 \
                 --run_eval True

# DTW Experiment - Wafer
# 39 Jobs
# python3 main.py --pname grad_replacement --task_id ${SLURM_ARRAY_TASK_ID} \
                 --run_mode turing --samples_per_task 4 \
                 --algo mse --dataset wafer --bbm dnn --bbm_path wafer_dnn_ce.ckpt \
                 --grad_replacement random \
                 --eval_replacement class \
                 --enable_dtw True --dtw_coeff 0.05 \
                 --background_data test --background_data_perc 100 \
                 --run_eval True

# Algo comparison Ford B Dataset (For threshold CM vs MSE)
# 15 Jobs
#python3 main.py --pname grad_replacement --task_id ${SLURM_ARRAY_TASK_ID} \
                 --run_mode turing --samples_per_task 54 \
                 --algo mse --dataset ford_b --bbm dnn --bbm_path ford_b_dnn_ce.ckpt \
                 --grad_replacement random \
                 --eval_replacement class \
                 --background_data test --background_data_perc 100 \
                 --run_eval True


# MSE Gradient replacement Experiments (Wafer)
# BD 100% 11 Jobs
#python3 main.py --pname grad_replacement --task_id ${SLURM_ARRAY_TASK_ID} \
                 --run_mode turing --samples_per_task 10 \
                 --algo shap --dataset wafer --bbm dnn --bbm_path wafer_dnn_ce.ckpt \
                 --grad_replacement random \
                 --eval_replacement class \
                 --background_data test --background_data_perc 100 \
                 --run_eval True


# Early Stopping Experiments
# python3 main.py --pname ford_a-nte-early-stop --task_id ${SLURM_ARRAY_TASK_ID} \
                 --run_mode single --single_sample_id 0 \
                 --algo mse  --dataset ford_a --bbm dnn --bbm_path ford_a_dnn_ce.ckpt \
                 --early_stopping False \
                 --run_eval_every_epoch True \
                 --max_itr 500


# Background Dataset experiments
# SHAP
#25 Jobs

#python3 main.py --pname bde-experiments --task_id ${SLURM_ARRAY_TASK_ID} \
                 --run_mode turing --samples_per_task 135 \
                 --algo shap  --dataset ford_a --bbm dnn --bbm_path ford_a_dnn_ce.ckpt \
                 --background_data none --background_data_perc 0 \
                 --run_eval True

#python3 main.py --pname bde-experiments --task_id ${SLURM_ARRAY_TASK_ID} \
                 --run_mode turing --samples_per_task 135 \
                 --algo shap  --dataset ford_a --bbm dnn --bbm_path ford_a_dnn_ce.ckpt \
                 --background_data test --background_data_perc 100 \
                 --run_eval True

#python3 main.py --pname bde-experiments --task_id ${SLURM_ARRAY_TASK_ID} \
                 --run_mode turing --samples_per_task 135 \
                 --algo shap  --dataset ford_a --bbm dnn --bbm_path ford_a_dnn_ce.ckpt \
                 --background_data test --background_data_perc 80 \
                 --run_eval True

#python3 main.py --pname bde-experiments --task_id ${SLURM_ARRAY_TASK_ID} \
                 --run_mode turing --samples_per_task 135 \
                 --algo shap  --dataset ford_a --bbm dnn --bbm_path ford_a_dnn_ce.ckpt \
                 --background_data test --background_data_perc 50


#python3 main.py --pname ford_a-nte-early-stop \
                 --run_mode turing --samples_per_task 60 \
                 --algo shap  --dataset ford_a --bbm dnn --bbm_path ford_a_dnn_ce.ckpt \
                 --background_data test background_data_perc 10 \
                 --run_eval True
