#!/usr/bin/env bash
# Input python command to be submitted as a job

#SBATCH --output=features-%j.out
#SBATCH --job-name features
#SBATCH --time=1:00:00
#SBATCH --mem=100000
#SBATCH -n 2

source globals.sh

module load anacondapy
source activate $CONDA_ENV

python ./scripts/$1 $SUBJ $PROJ_DIR 
