#!/bin/bash
#
# analyze.sh runs the analysis of a subject
# original author: mason simon (mgsimon@princeton.edu)
# this script was provided by NeuroPipe. modify it to suit your needs


#SBATCH -J 'flirtff'
#SBATCH -o slurm-%j.out
#SBATCH -p all
#SBATCH -t 5:00:00
#SBATCH --export=globals.sh

set -e # stop immediately when an error occurs


pushd $SLURM_SUBMIT_DIR > /dev/null   # move into the subject's directory, quietly

echo $(dirname $0)
echo $SLURM_SUBMIT_DIR
echo "In the directory: `pwd` "
echo "As the user: `whoami` "
echo "on host: `hostname` "

echo "With access to cpu id(s): "
cat /proc/$$/status | grep Cpus_allowed_list

echo "Array Allocation Number: $SLURM_ARRAY_JOB_ID"
echo "Array Index: $SLURM_ARRAY_TASK_ID"


CURR_DIR=$SLURM_SUBMIT_DIR

source $CURR_DIR/globals.sh # load subject-wide settings


### Which numbered runs are being processed? Delineate below in run list. 

drawList="1 2 3 4"
recList="1 2 3 4 5 6"


echo "== beginning conversion of $SUBJ at $(date) =="

### Use flirt to register the parameter estimates and copes to the subjects own T1 anatomical

if [[ ! -d "$FIRSTLEVEL_DIR/parameter" ]]; then
    mkdir $FIRSTLEVEL_DIR/parameter
fi


HIRES=$NIFTI_DIR/${SUBJ}_anat_mprage_brain.nii.gz
for r in $drawList
do
     FF=$FIRSTLEVEL_DIR/draw_run_${r}.feat/filtered_func_data.nii.gz
     FUNC2HIRES=$FIRSTLEVEL_DIR/draw_run_${r}.feat/reg/example_func2highres.mat
     OUTFILE=$FIRSTLEVEL_DIR/parameter/${SUBJ}_draw_run_${r}_filtfuncHIRES.nii.gz
     flirt -ref $HIRES  -in $FF -applyisoxfm 2 -init $FUNC2HIRES -out $OUTFILE -interp spline
     echo "drawing run ${r} in T1 space -- $(date)"
done

for r in $recList
do
     FF=$FIRSTLEVEL_DIR/glm4_recognition_run_${r}.feat/filtered_func_data.nii.gz
     FUNC2HIRES=$FIRSTLEVEL_DIR/glm4_recognition_run_${r}.feat/reg/example_func2highres.mat
     OUTFILE=$FIRSTLEVEL_DIR/parameter/${SUBJ}_recog_run_${r}_filtfuncHIRES.nii.gz
     flirt -ref $HIRES  -in $FF -applyisoxfm 2 -init $FUNC2HIRES -out $OUTFILE -interp spline
     echo "recog run ${r} in T1 space -- $(date)"
done




popd > /dev/null   # return to the directory this script was run from, quietly
