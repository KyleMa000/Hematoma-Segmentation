#!/bin/bash
# The interpreter used to execute the script

# directives that convey submission options:

#SBATCH --job-name=hematoma_segmentation
#SBATCH --mail-user=#####
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --account=#####
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load python/3.10.4
module load pytorch
module load matplotlib
pip install -U tensorly
pip install -U tensorly-torch
# The application(s) to execute along with its input arguments and options:
python train.py