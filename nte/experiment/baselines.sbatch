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
## CricketX

# BaseLine Experiments
# 16 Jobs, 10 SPT
python3 -u main.py --pname baselines --run_id 4 --task_id ${SLURM_ARRAY_TASK_ID} \
                 --run_mode turing --samples_per_task 10 \
                 --algo shap --dataset wafer --bbm dnn --bbm_path Baseline_WaferDataset_model_dnn_loss_ce_lr_0.0005_epochs_120.ckpt \
                 --grad_replacement zeros \
                 --eval_replacement class \
                 --enable_dtw False --dtw_coeff 0.05 \
                 --background_data test --background_data_perc 100 \
                 --run_eval True
