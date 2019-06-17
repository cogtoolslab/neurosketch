#!/bin/bash
#
#SBATCH -J 'flirt'
#SBATCH -o flirt-%j.out
#SBATCH -p all
#SBATCH -t 5:00:00
#SBATCH --export=globals.sh

#set -e # stop immediately when an error occurs


pushd $SLURM_SUBMIT_DIR > /dev/null   # move into the subject's directory, quietly

CURR_DIR=$SLURM_SUBMIT_DIR
FL_DIR=$CURR_DIR/analysis/firstlevel
source $CURR_DIR/globals.sh # load subject-wide settings
STANDARD=$CURR_DIR/../../standard/MNI152_T1_2mm_brain

### Which runs are being processed? Delineate below in run list. 

runList="glm4_recognition_run_1 glm4_recognition_run_2 glm4_recognition_run_3 \
         glm4_recognition_run_4 glm4_recognition_run_5 glm4_recognition_run_6 \
         draw_run_1, draw_run_2, draw_run_3, draw_run_4"
copeList="1 2 3 4"

echo "== beginning recognition runs for $SUBJ at $(date) =="

for r in $runList
do
  HIRES=$NIFTI_DIR/${SUBJ}_anat_mprage_brain.nii.gz
  FF=$FL_DIR/${r}.feat/filtered_func_data.nii.gz
  OUTFF=$FL_DIR/${r}.feat/filtfuncHIRES.nii.gz
  ff=$FL_DIR/${r}.feat/reg/example_func2highres.nii.gz
  outff=$FL_DIR/${r}.feat/reg/func2highres2mm.nii.gz
  MAT=$FL_DIR/${r}.feat/reg/example_func2highres.mat
  STAN=$FL_DIR/${r}.feat/reg/func2standard.mat
  flirt -ref $HIRES  -in $FF -applyisoxfm 2 -out $OUTFF -interp spline
  flirt -ref $HIRES  -in $ff -applyisoxfm 2 -out $outff -init $MAT -interp spline
  
  if [[ ${r:0:4} != "draw" ]]; 
  then
    for cope in $copeList
    do
      INCOPE=$FL_DIR/${r}.feat/stats/cope${cope}.nii.gz
      OUTCOPE=$FL_DIR/${r}.feat/stats/cope${cope}_HIRES.nii.gz
      STANCOPE=$FL_DIR/${r}.feat/stats/cope${cope}_STAN.nii.gz
      flirt -ref $HIRES  -in $INCOPE -applyisoxfm 2 -out $OUTCOPE -init $MAT -interp spline
      flirt -ref $STANDARD  -in $INCOPE -applyisoxfm 2 -out $STANCOPE -init $STAN -interp spline
    done
  fi
  echo "run ${r} in T1 space -- $(date)"
done


echo "== finished converting to T1 space for $SUBJ at $(date) =="


popd > /dev/null   # return to the directory this script was run from, quietly

