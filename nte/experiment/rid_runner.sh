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


# Wafer Best - 32, Least - 134
# Iterating through an array with index
for i in {0..151}
do
   echo "WORKING ON $i"
   python3 main.py --pname rindex_exp_v1 --run_mode single \
                 --algo nte --dataset wafer \
                 --enable_dist False --dist_loss no_dist --dist_coeff 0 \
                 --grad_replacement random_instance \
                 --eval_replacement class_mean \
                 --background_data test --background_data_perc 100 \
                 --run_eval True \
                 --r_index $i --dynamic_replacement False --enable_seed True --seed_value 0 \
                 --run 1 \
                 --single_sample_id ${SLURM_ARRAY_TASK_ID} \
                 --max_itr 500
done
