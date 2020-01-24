#!/bin/bash
#
# analyze.sh runs the analysis of a subject
# original author: mason simon (mgsimon@princeton.edu)
# this script was provided by NeuroPipe. modify it to suit your needs


#SBATCH -J 'connect'
#SBATCH -o slurm-%j.out
#SBATCH -p all
#SBATCH -t 10:00:00
#SBATCH --mem=25GB
#SBATCH -n 2
#SBATCH --export=globals.sh

#set -e # stop immediately when an error occurs

source globals.sh
module load anacondapy/3.4
source activate $CONDA_ENV

echo $SLURM_SUBMIT_DIR
pushd $SLURM_SUBMIT_DIR > /dev/null

if [ ! -d group ]; then
  mkdir group
fi


python -u scripts/$1 $2
