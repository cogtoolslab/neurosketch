#!/bin/bash
#
# analyze.sh runs the analysis of a subject
# original author: mason simon (mgsimon@princeton.edu)
# this script was provided by NeuroPipe. modify it to suit your needs


#SBATCH -J 'connect'
#SBATCH -o slurm-%j.out
#SBATCH -p all
#SBATCH -t 3:00:00
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


subjectList="0110171_neurosketch 0113171_neurosketch 0119172_neurosketch 0123171_neurosketch 1130161_neurosketch 1207162_neurosketch 0110172_neurosketch 0115174_neurosketch 0119173_neurosketch 0123173_neurosketch 1202161_neurosketch 0111171_neurosketch 0117171_neurosketch 0119174_neurosketch 0124171_neurosketch 1203161_neurosketch 0112171_neurosketch 0118171_neurosketch 0120171_neurosketch 0125171_neurosketch 1206161_neurosketch 0112172_neurosketch 0118172_neurosketch 0120172_neurosketch 0125172_neurosketch 1206162_neurosketch 0112173_neurosketch 0119171_neurosketch 0120173_neurosketch 1121161_neurosketch 1206163_neurosketch"
# produce list of all subjects but the current subject
# make fsf file (name = subject) for that higher level analysis (similar to render_secondlevel
# run feat (output dir = subject)
for subject in $subjectList
do
  echo "subject started -- $(date)"
  delete=($subject)
  allothers=("${subjectList[@]/$delete}")
  echo "${subject} --- making template"
#  bash scripts/make_template.sh "${allothers[@]}" $subject
  echo "${subject} --- running feat"
#  feat group/${subject}.fsf &
#  sleep 5
done

#sleep 250
# wait for feats to finish
# make mask (intersect of univariate mask with freesurfer ROIs)
# make features for the given subject
for subject in $subjectList
do
  pushd $SLURM_SUBMIT_DIR > /dev/null
  echo "${subject} --- waiting for feat -- $(date)"
  bash scripts/wait-for-feat.sh group/${subject}.gfeat
#  cp group/${subject}.gfeat/cope1.feat/thresh_zstat1.nii.gz group/${subject}.gfeat/draw_task_mask.nii.gz
  bash scripts/binarize_hires.sh $subject
  echo "${subject} --- masks made -- $(date)"
  pushd subjects/$subject >/dev/null
  sbatch scripts/features_metadata.sh draw_features.py 9
  echo "${subject} --- generating features for intersect rois -- $(date)"
done

for subject in $subjectList
do
  pushd subjects/$subject >/dev/null
  subj=$(echo ${subject} | cut -d"_" -f1)
  while [ ! -e "${PROJECT_DIR}/../../data/features/production/${subj}_ParietalDraw_featurematrix.npy" ]; do
    sleep 20
    echo "waiting for draw features"
  done
  echo "${subject} --- generating connectivity features -- $(date)"
  sbatch scripts/features_metadata.sh connect_features.py
  popd >/dev/null
done

for subject in $subjectList
do
  subj=$(echo ${subject} | cut -d"_" -f1)
  while [ ! -e "${PROJECT_DIR}/../../data/features/connectivity/${subj}_LOCDraw_ParietalDraw_stackmatrix.npy" ]; do
  echo "waiting for ${subject} connectivity features to be finished"
#  while [ ! -e "${PROJECT_DIR}/../../data/features/connectivity/${subj}_PRC_ento_stackmatrix.npy" ]; do
    sleep 1
  done
  echo "${subject} connectivity features finished"
done
echo "FINISHED"
