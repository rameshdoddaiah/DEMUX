#!/bin/bash
#SBATCH -p short
#SBATCH -N 1                      # number of nodes
#SBATCH -n 4                      # number of cores
#SBATCH --mem=128GB                # memory pool for all cores
#SBATCH -t 0-24:00                # time (D-HH:MM)
#SBATCH --checkpoint=5
#SBATCH --checkpoint-dir=checkpoints
#SBATCH --gres=gpu:0              # number of GPU
#SBATCH --job-name=main
#SBATCH -o logs/slurm-main-output_%A-%a    # STDOUT
#SBATCH -e logs/slurm-main-error_%A-%a     # STDERR
#SBATCH --mail-type=ALL
#SBATCH --mail-user=rdoddaiah@wpi.edu
##SBATCH --exclude=compute-0-27
##SBATCH -C T4

#python3 main.py mse random_eval.json /home/psparvatharaju/work/Git/imagen/imagen ${SLURM_ARRAY_TASK_ID} --blur=True --tvnorm=False

#python3 main.py cm random_eval.json /home/psparvatharaju/work/Git/imagen/imagen ${SLURM_ARRAY_TASK_ID} --blur=True --tvnorm=True

######python3 sync.py

#nte
#mse
#cm
#rise
#shap
#lime

## Dataset Configuration
## Wafer (152) | spt 2 | j 76
## Ford B (810) | spt 9 | j 90
## Ford A (1320) | spt 15 | j 88
## Earthquake (139) | spt 2 | j 70
## GunPoint (150) | spt 2 | j 75
## CricketX (61)
## Computers (250) 

#Order
# number_to_dataset = {
#     "1": "wafer",
#     "2": "cricket_x",
#     "3": "gun_point",
#     "4": "earthquakes",
#     "5": "computers",
#     "6": "ford_a",
#     "7": "ford_b"
#     "8": "ptb"
#     "9": "ecg"
#     "10":"CricketX-MC"
# }

#1 normal default param values
#2 p-nte-4 default param values
#3 no noise p-nte-4
#4 mse coefficient zero on p-nte-4
#5 l1_coeff zero on p-nte-4
#6 tv_coeff zero on p-nte-4

#random
#mse
#nte
#p-nte-v4
#lime
#rise
#shap
#cm 
#rise-rep
#  --eval_replacement_mc min_class_random_index\
#  --algo pert-mc --dataset blip-mc \
#  --algo pert-mc \ 
#  --algo rise \ 
#  --algo shap \ 
#  --algo cm \ 
#  --algo lime \ 
#  --dataset cricketx-mc \
#  --dataset ACSF\
#  --dataset AbnormalHeartbeat\
#  --dataset EOGHorizontalSignal-mc\
# samples_per_task is number of threads
# pname hearbeat_mc, dataset AbnormalHeartbeat jobs_per_task 30
# pname trace, Trace-mc jobs_per_task 10
# max_itr or seed value change to get different std dev
#LMUX default params ssd dynamic, p_buf_init cluster, algo pert-mc, mse yes, kl yes, new_auc yes,  old_auc no max_itr 5000

#1 lime
#2 rise
#3 rise-rep
#4 CM
#5 PERT Not done
#6 LMUX Not done
#7 SHAP
#ACSF done; Plane-mc
#Final_MUX_DNN and LMUX

python3 main.py --pname newICDM --task_id ${SLURM_ARRAY_TASK_ID} \
                 --run_mode turing --jobs_per_task 10\
                 --samples_per_task 10\
                 --dataset Trace-mc\
                 --algo timeshap\
                 --p_buf_init cluster\
                 --seed_value 26\
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
                 --bbm rnn \
                 --eval_replacement_mc cluster_mean\
                 --multi_class yes\
                 --mse yes\
                 --ce no\
                 --ml no\
                 --kl yes\
                 --num_perturbs 500\
                 --class_prob 0.0\
                 --mask_mse_coeff 1.0\
                 --mse_coeff 1.0\
                 --enable_tvnorm True\
                 --enable_budget True\
                 --print no\
                 --top_n yes\
                 --dataset_type test\
                 --clusters mean\
                 --num_masks 5000\
                 --l_prev kld\
                 --l_budget_coeff 0.6\
                 --run 1\
                 --l_ssd_coeff 0.01\
                 --l_tv_norm_coeff 0.5\
                 --iou_alpha 0.5\
                 --l_prev_coeff 1.0\
                 --l_max_coeff 0.7\



#Computers 
#python3 main.py --algo rise --run 1 --pname Paper --dataset_type test --run_mode turing --dataset computers   --bbm dnn --samples_per_task 15  --task_id ${SLURM_ARRAY_TASK_ID} --bbm_path computers_dnn_ce.ckpt  

#Gunpoint
#python3 main.py --algo cmwr --run 31 --pname cm_replacement --dataset_type test --run_mode turing --dataset gun_point   --bbm dnn --samples_per_task 15  --task_id ${SLURM_ARRAY_TASK_ID} --bbm_path gun_point_dnn_ce.ckpt  

#Earthquake
#python3 main.py --algo cm --run 1 --pname Paper --dataset_type test --run_mode turing --dataset earthquakes --bbm dnn --samples_per_task 15  --task_id ${SLURM_ARRAY_TASK_ID} --bbm_path earthquakes_dnn_ce.ckpt  

#Ford A
#python3 main.py --algo lime --run 1 --pname Paper --dataset_type test --run_mode turing --dataset ford_a      --bbm dnn --samples_per_task 130 --task_id ${SLURM_ARRAY_TASK_ID} --bbm_path ford_a_dnn_ce.ckpt  

#Ford B
#python3 main.py --algo rise --run 1 --pname Paper --dataset_type test --run_mode turing --dataset ford_b      --bbm dnn --samples_per_task 80  --task_id ${SLURM_ARRAY_TASK_ID} --bbm_path ford_b_dnn_ce2.ckpt  

#Wafer
#python3 main.py --algo cmwr --run 32 --pname cm_replacement --dataset_type test --run_mode turing --dataset wafer       --bbm dnn --samples_per_task 13  --task_id ${SLURM_ARRAY_TASK_ID} --bbm_path wafer_dnn_ce.ckpt  

#CricketX
#python3 main.py --algo cm --run 1 --pname Paper --dataset_type test --run_mode turing --dataset cricket_x   --bbm dnn --samples_per_task 39  --task_id ${SLURM_ARRAY_TASK_ID} --bbm_path cricket_x_dnn_ce.ckpt  
