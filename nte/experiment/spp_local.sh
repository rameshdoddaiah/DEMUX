#!/bin/bash

export PYTHONPATH=$PYTHONPATH:/Users/prathyushsp/Git/TimeSeriesSaliencyMaps/:/Users/prathyushsp/Git/TimeSeriesSaliencyMaps/nte/models/saliency_model/tsmule/ts_mule:/Users/prathyushsp/Git/TimeSeriesSaliencyMaps/nte/models/saliency_model/timeshap/src

# python3 main.py --pname baselines --run_mode local \
#                  --algo rise-rep --dataset computers --bbm dnn --bbm_path computers_dnn_ce.ckpt \
#                  --grad_replacement random_instance \
#                  --eval_replacement class_mean \
#                  --background_data test --background_data_perc 100 \
#                  --run_eval True \
#                  --run_id 1



# python3 main.py --pname ACSF_distinct2 --run_mode single \
#                  --dataset ACSF\
#                  --algo pert-mc\
#                  --run 140\
#                  --class_prob 1.0\
#                  --seed_value 1\
#                  --enable_dist False --dist_loss no_dist --dist_coeff 0 \
#                  --enable_lr_decay False \
#                  --grad_replacement random_instance \
#                  --eval_replacement class_mean \
#                  --background_data test --background_data_perc 100\
#                  --run_eval True  --enable_seed True \
#                  --max_itr 100\
#                  --ssd dynamic\
#                  --run_id 1 \
#                  --w_decay 0.00 \
#                  --bbm dnn \
#                  --eval_replacement_mc min_class_mean\
#                  --multi_class yes\
#                  --mse yes\
#                  --first_class no\
#                  --second_class no\
#                  --third_class no\
#                  --ce no\
#                  --ml no\
#                  --kl no\
#                  --num_perturbs 500\
#                  --prob_upto 0.0 \
#                  --early_stop_prob_range 0.05 \
#                  --early_stop_diff 0\
#                  --early_stop_min_epochs 0\
#                  --early_stopping False\
#                  --tsne no \
#                  --mask_mse_coeff 1\
#                  --old_auc no\
#                  --new_auc yes\
#                  --top_class yes\
#                  --enable_hinge_loss no\
#                  --enable_triplet_loss no\
#                  --preserve_prob no\
#                  --mse_coeff 1.0\
#                  --enable_tvnorm False\
#                  --enable_budget False\
#                  --print yes\
#                  --top_n yes\
#                  --dataset_type test\
#                  --sample_masks 1\
#                  --clusters mean\
#                  --n_clusters 1\
#                  --cluster_indx 0\
#                  --cluster_dtw dtw\
#                  --bbmg no\
#                  --window_size 1\
#                  --nth_prob 0.0\


# AbnormalHeartbeat, ACSF
# python3 main.py --pname Baseline_MC --run_mode single \
#                  --dataset ACSF\
#                  --algo tsmule\
#                  --run 140\
#                  --class_prob 1.0\
#                  --seed_value 1\
#                  --enable_dist False --dist_loss no_dist --dist_coeff 0 \
#                  --enable_lr_decay False \
#                  --grad_replacement random_instance \
#                  --eval_replacement class_mean \
#                  --background_data test --background_data_perc 100\
#                  --run_eval True  --enable_seed True \
#                  --max_itr 100\

#                  --run_id 1 \

#                  --bbm dnn \
#                  --eval_replacement_mc cluster_mean\
#                  --multi_class yes\

#                  --first_class no\
#                  --second_class no\
#                  --third_class no\
#                  --ce no\
#                  --ml no\
#                  --kl yes\
#                  --num_perturbs 500\
#                  --prob_upto 0.0 \
#                  --early_stop_prob_range 0.05 \
#                  --early_stop_diff 0\
#                  --early_stop_min_epochs 0\
#                  --early_stopping False\
#                  --tsne no \
#                  --mask_mse_coeff 0.1\
#                  --old_auc no\
#                  --new_auc yes\
#                  --top_class yes\
#                  --enable_hinge_loss no\
#                  --enable_triplet_loss no\
#                  --preserve_prob no\
#                  --mse_coeff 1.0\
#                  --enable_tvnorm False\
#                  --enable_budget False\
#                  --print yes\
#                  --top_n yes\
#                  --dataset_type test\
#                  --sample_masks 1\
#                  --clusters mean\
#                  --n_clusters 1\
#                  --cluster_indx 0\
#                  --cluster_dtw dtw\
#                  --bbmg no\
#                  --window_size 1\
#                  --nth_prob 0.0\
#                  --prob_loss yes\



python3 main.py --pname Baseline_MC --run_mode single \
                 --dataset ACSF\
                 --algo timeshap\
                 --run 140\
                 --seed_value 1\
                 --grad_replacement random_instance \
                 --eval_replacement class_mean \
                 --background_data test --background_data_perc 100\
                 --run_eval True  --enable_seed True \
                 --run_id 1 \
                 --bbm dnn \
                 --eval_replacement_mc cluster_mean\
                 --multi_class yes\
