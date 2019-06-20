#!/bin/bash

#SBATCH -J 'convert'
#SBATCH -o slurm-%j.out
#SBATCH -p all
#SBATCH -t 90:00
#SBATCH --export=globals.sh

#set -e # stop immediately when an error occurs


pushd $SLURM_SUBMIT_DIR > /dev/null   # move into the subject's directory, quietly


CURR_DIR=$SLURM_SUBMIT_DIR
source $CURR_DIR/globals.sh # load subject-wide settings

HIRES=$NIFTI_DIR/${SUBJ}_anat_mprage.nii.gz

if [ ! -d "$OUTPUT_DIR" ]; then
  mkdir $OUTPUT_DIR
fi

runList="1 2 3 4 5 6"
copeList="$(seq -s ' ' 1 4)"

### Use flirt to register copes from original directory to the subjects own T1 anatomical

for r in $runList
do
    FUNC2HIRES=$FIRSTLEVEL_DIR/glm4_recognition_run_${r}.feat/reg/example_func2highres.mat
    ORIGINAL=$FIRSTLEVEL_DIR/glm4_recognition_run_${r}.feat/example_func.nii.gz
    for cope in $copeList
    do
         COPE=$FIRSTLEVEL_DIR/glm4_recognition_run_${r}.feat/stats/cope${cope}.nii.gz
         OUTFILE=$OUTPUT_DIR/${SUBJ}_recognition_run_${r}_cope${cope}.nii.gz
         flirt -ref $HIRES  -in $COPE -applyisoxfm 2 -init $FUNC2HIRES -out $OUTFILE -interp spline
         echo "run ${r}, cope ${cope} in T1 space -- $(date)"
    done
done

flirt -in $ORIGINAL -ref $HIRES -applyisoxfm 2 -init $FUNC2HIRES -out $OUTPUT_DIR/2mmExample.nii.gz -interp spline


echo "== finished converting copes for $SUBJ at $(date) =="
