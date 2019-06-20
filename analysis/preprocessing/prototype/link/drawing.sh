#!/bin/bash
#SBATCH -J 'fullSub'
#SBATCH -o fullSub-%j.out
#SBATCH -p all
#SBATCH -t 24:00:00
#SBATCH -c 2
#SBATCH --mem=20GB
#SBATCH --export=globals.sh

set -e # stop immediately when an error occurs

module load anacondapy/2.7

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
SUBJECTS_DIR=$CURR_DIR/analysis/firstlevel
LEVEL2=$CURR_DIR/analysis/secondlevel

source $CURR_DIR/globals.sh # load subject-wide settings

### This script first runs the preprocessing (convert dicom to nifti, qa data, reorient to LAS)
### Next, it will render the templates necessary for the feats, and brain extract the images
### It then waits for all of the feats to be done, before moving on
### Next, it takes the functional time series (filtered_func_data) and transforms it to subject's anatomical space.

### Which numbered runs are being processed? Delineate below in run list. 

drawList="1 2 3 4"
recList="1 2 3 4 5 6"

### Initial basic preprocessing

echo "== beginning analysis of $SUBJ at $(date) ==" | tee status

bash prep.sh
echo "i. prep done -- $(date)" | tee status
bash scripts/render_fsf_templates_draw.sh
bash scripts/render_fsf_templates_glm4.sh
echo "ii. fsf templates done -- $(date)" | tee status
bash brain_extract_first.sh | tee status
echo "iii. brain extraction done -- $(date)" | tee status

### Run freesurfer on the anatomical to extract ROIs

ANAT=$CURR_DIR/$NIFTI_DIR/${SUBJ}_anat_mprage.nii.gz
#recon-all -all -notify $FIRSTLEVEL_DIR/freesurfer/surfdone -subjid freesurfer -i $ANAT &> /dev/null &
echo "iv. freesurfer started -- $(date)" | tee status


### Run feat for each template generated in the above scripts

for r in $drawList
do
     feat $FSF_DIR/draw_run_${r}.fsf &
     echo "drawing run ${r} feat started -- $(date)" | tee status
     sleep 30
done

for r in $recList
do
     feat $FSF_DIR/glm4_recognition_run_${r}.fsf &
     echo "recognition run ${r} feat started -- $(date)" | tee status
     sleep 30
done
echo "v. all feats started -- $(date)" | tee status


### Wait for feats to be done for each run before moving on to the next step.

for r in $drawList
do
    bash scripts/wait-for-feat.sh $FIRSTLEVEL_DIR/draw_run_${r}.feat
done

for r in $recList
do
    bash scripts/wait-for-feat.sh $FIRSTLEVEL_DIR/glm4_recognition_run_${r}.feat
done

echo "vi. all level1 feats done -- $(date)" | tee status


# render the template for higher order drawing run analysis
function render_secondlevel {
  fsf_template=$1
  output_dir=$2
  standard_brain=$3
  first_level_dir=$4


  cat $fsf_template \
    | sed "s:<?= \$OUTPUT_DIR ?>:$output_dir:g" \
    | sed "s:<?= \$STANDARD_BRAIN ?>:$standard_brain:g" \
    | sed "s:<?= \$FIRST_DIR ?>:$first_level_dir:g"
}

render_secondlevel $FSF_DIR/draw_level2.fsf.template \
                   $LEVEL2/draw \
                   /jukebox/pkgs/FSL/5.0.9/data/standard/MNI152_T1_2mm_brain \
                   $SUBJECTS_DIR \
                   > $FSF_DIR/draw_level2.fsf

sleep 10

# Run higher order drawing run analysis
feat $FSF_DIR/draw_level2.fsf
echo "vii. Running second level analysis -- $(date)" | tee status

while [ ! -d "${LEVEL2}/draw.gfeat" ]; do
  sleep 1
done

# Move reg files over
cp -R $SUBJECTS_DIR/draw_run_1.feat/reg analysis/secondlevel/draw.gfeat
#cp $SUBJECTS_DIR/draw_run_1.feat/example_func.nii.gz analysis/secondlevel/draw.gfeat


# convert copes and filtered func data to subject anatomical space
echo "viii. Running conversion scripts for copes and filtered func -- $(date)" | tee status
sbatch scripts/convert_copes.sh
sbatch scripts/flirt_filt.sh
echo "submitted jobs -- $(date)" | tee status

### Wait for freesurfer to finish
mv $FIRSTLEVEL_DIR/register.dat $FIRSTLEVEL_DIR/freesurfer/register.dat
mv $FIRSTLEVEL_DIR/surfdone $FIRSTLEVEL_DIR/freesurfer/surfdone
bash scripts/wait-for-surf.sh .
echo "ix. freesurfer done -- $(date)" | tee status


while [ ! -e "${FIRSTLEVEL_DIR}/parameter/2mmExample.nii.gz" ]; do
  echo "conversion not yet complete"
  sleep 1
done

echo "x. Creating surfer ROIs -- $(date)" | tee status
bash scripts/surfROI.sh
echo "x. Surfer ROIs done -- $(date)" | tee status

echo "xi. Creating features -- $(date)" | tee status
sbatch scripts/features_metadata.sh recog_features.py
sbatch scripts/features_metadata.sh draw_features.py

subj=$(echo ${SUBJ} | cut -d"_" -f1)
while [ ! -e "${PROJ_DIR}/data/features/production/${subj}_ParietalDraw_featurematrix.npy" ]; do
  sleep 20
done
while [ ! -e "${PROJ_DIR}/data/features/recognition/metadata_${subj}_hipp_56.csv" ]; do
  sleep 20
done
echo "xi. Features done! -- $(date)" | tee status

# SUBJECT IS DONE!
echo "== finished analysis of $SUBJ at $(date) ==" | tee status

popd > /dev/null   # return to the directory this script was run from, quietly
