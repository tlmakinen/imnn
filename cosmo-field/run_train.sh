#!/bin/bash
#SBATCH -p gpu
#SBATCH --job-name=field_imnn
#SBATCH --array=1-4
#SBATCH --output=./slurmscripts/job_%A_%a.out
#SBATCH --nodes=1
# SBATCH --ntasks=4
#SBATCH --cpus-per-task=10
#SBATCH --time=0-3:00:00
#SBATCH --gres=gpu:v100-32gb:1
#SBATCH --mem=730gb

module load  gcc/7.4.0 cuda/10.1.243_418.87.00 cudnn/v7.6.2-cuda-10.1 nccl/2.4.2-cuda-10.1 python3/3.7.3
# copy code from repository into jobfile
source ~/anaconda3/bin/activate pyimnn
python3 field_run.py $SLURM_ARRAY_TASK_ID 
