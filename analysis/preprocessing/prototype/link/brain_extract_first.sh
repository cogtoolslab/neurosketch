#!/bin/bash
#
# analyze.sh runs the analysis of a subject
# original author: mason simon (mgsimon@princeton.edu)
# this script was provided by NeuroPipe. modify it to suit your needs


#SBATCH -J 'bet'
#SBATCH -o slurm-%j.out
#SBATCH -p all
#SBATCH -t 00:15:00
#SBATCH --export=globals.sh

set -e # stop immediately when an error occurs

pushd $SLURM_SUBMIT_DIR > /dev/null   # move into the subject's directory, quietly
NOW=$SLURM_SUBMIT_DIR
source $NOW/globals.sh

echo $(date)
echo $(dirname $0)
echo $SLURM_SUBMIT_DIR


bet $NIFTI_DIR/${SUBJ}_anat_mprage.nii.gz $NIFTI_DIR/${SUBJ}_anat_mprage_brain.nii.gz
