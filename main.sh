#!/usr/bin/env bash
#SBATCH --partition gpu
#SBATCH --time 0:30:00
#SBATCH --mem 64GB
#SBATCH --gres gpu:1
#SBATCH --job-name clean_data
#SBATCH --nodes 1
#SBATCH --ntasks-per-node 28

# get rid of any modules already loaded
module purge
# load in the module dependencies for this script
module load "languages/anaconda3/2019.07-3.6.5-tflow-1.14"

python ./processing_functions.py -a -q pitch_control -k optimal -s optimal -n 28 -m 1.0
