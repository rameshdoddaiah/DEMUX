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

#python3 main.py mse random_eval.json /home/psparvatharaju/work/Git/imagen/imagen ${SLURM_ARRAY_TASK_ID} --blur=True --tvnorm=False

#python3 main.py cm random_eval.json /home/psparvatharaju/work/Git/imagen/imagen ${SLURM_ARRAY_TASK_ID} --blur=True --tvnorm=True

python3 sync.py
