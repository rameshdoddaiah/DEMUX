#!/bin/bash
################################
# Script for running multiple experiments by submitting multiple slurm jobs
# How to use this script?
#  ./main.sh [taskid range]
# For example, if you want to run task id 1 to id 5 (5 tasks in parallel), you could type:
# ./main.sh 1-5
################################
# sbatch --array=$1 main.sbatch

#srun --array=$1 main.sbatch

# TURING
#export PYTHONPATH=/home/psparvatharaju/work/xai/bin/python3
#source /home/psparvatharaju/work/xai/bin/activate

#export PYTHONPATH=/home/rdoddaiah/anaconda3/envs/pytorch/bin/python
#source /home/rdoddaiah/anaconda3/bin/activate


export PYTHONPATH=/home/rdoddaiah/work/TimeSeriesSaliencyMaps
export PYTHONPATH=$PYTHONPATH:/home/rdoddaiah/work/TimeSeriesSaliencyMaps/nte/models/saliency_model/tsmule/ts_mule:/home/rdoddaiah/work/TimeSeriesSaliencyMaps
source /home/rdoddaiah/anaconda3/bin/activate pytorch

sbatch --array=$1 rd_main.sbatch

# LOCAL
#source /home/tom/Documents/workenv/bin/activate
#python run_models.py --taskid=$1

################################
# How to use more GPUs?
# change in file: main.sbatch
################################
