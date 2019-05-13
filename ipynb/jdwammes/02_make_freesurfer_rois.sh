#!/bin/bash
#
# Script that creates .nii masks based on Freesurfer .label files.


#SBATCH -J 'srufROIs'
#SBATCH -o slurm-%j.out
#SBATCH -p all
#SBATCH -t 10:00
#SBATCH --export=globals.sh

#set -e # stop immediately when an error occurs


pushd $SLURM_SUBMIT_DIR > /dev/null   # move into the subject's directory, quietly

CURR_DIR=$SLURM_SUBMIT_DIR
FL_DIR=$CURR_DIR/analysis/firstlevel
SUBJECT=freesurfer

source $CURR_DIR/globals.sh # load subject-wide settings


ROI_DIR=$SUBJECTS_DIR/surfROI


if [ ! -d "$ROI_DIR" ]; then
  mkdir $ROI_DIR
fi

FILLTHRESH=0.0 # threshold for incorporating a voxel in the mask, default = 0
REFERENCEFUNC=$FL_DIR/${r}.feat/reg/func2highres2mm.nii.gz
REFERENCEVOLUME=${CURR_DIR}/data/nifti/${SUBJ}_anat_mprage_brain.nii.gz

ROIS="V1 V2 perirhinal"

for ROI in $ROIS
do
  echo $ROI
  for HEMI in lh rh
  do
    LABEL=${FL_DIR}/freesurfer/label/${HEMI}.${ROI}_exvivo.label
    DEST=${ROI_DIR}/${HEMI}.${ROI}.nii.gz
    mri_label2vol --label $LABEL --temp $REFERENCEFUNC --o $DEST --fillthresh \
    $FILLTHRESH --proj frac 0 1 0.1 --subject $SUBJECT --hemi $HEMI --surf white
  done
done


mri_convert -rl $REFERENCEFUNC -rt nearest ${SUBJECTS_DIR}/freesurfer/mri/aparc+aseg.mgz ${ROI_DIR}/aparc+aseg.nii.gz


# names of labels can be looked up in $FREESURFER_HOME/FreeSurferColorLUT.txt

# left IT = 1009, right IT = 2009
fslmaths ${ROI_DIR}/aparc+aseg.nii.gz -uthr 1009 -thr 1009 ${ROI_DIR}/lh.IT_FS.nii.gz
fslmaths ${ROI_DIR}/aparc+aseg.nii.gz -uthr 2009 -thr 2009 ${ROI_DIR}/rh.IT_FS.nii.gz

# left hippocampus = 17, right hippocampus = 53
fslmaths ${ROI_DIR}/aparc+aseg.nii.gz -uthr 17 -thr 17 ${ROI_DIR}/lh.hipp_FS.nii.gz
fslmaths ${ROI_DIR}/aparc+aseg.nii.gz -uthr 53 -thr 53 ${ROI_DIR}/rh.hipp_FS.nii.gz

# left entorhinal = 1006, right entorhinal = 2006
fslmaths ${ROI_DIR}/aparc+aseg.nii.gz -uthr 1006 -thr 1006 ${ROI_DIR}/lh.ento_FS.nii.gz
fslmaths ${ROI_DIR}/aparc+aseg.nii.gz -uthr 2006 -thr 2006 ${ROI_DIR}/rh.ento_FS.nii.gz

# left parahippocampal = 1016, right parahippocampal = 2016
fslmaths ${ROI_DIR}/aparc+aseg.nii.gz -uthr 1016 -thr 1016 ${ROI_DIR}/lh.parahippo_FS.nii.gz
fslmaths ${ROI_DIR}/aparc+aseg.nii.gz -uthr 2016 -thr 2016 ${ROI_DIR}/rh.parahippo_FS.nii.gz

# left lateraloccipital = 1011, right lateraloccipital = 2011
fslmaths ${ROI_DIR}/aparc+aseg.nii.gz -uthr 1011 -thr 1011 ${ROI_DIR}/lh.LOC_FS.nii.gz
fslmaths ${ROI_DIR}/aparc+aseg.nii.gz -uthr 2011 -thr 2011 ${ROI_DIR}/rh.LOC_FS.nii.gz

# left fusiform = 1007, right fusiform = 2007
fslmaths ${ROI_DIR}/aparc+aseg.nii.gz -uthr 1007 -thr 1007 ${ROI_DIR}/lh.fusiform_FS.nii.gz
fslmaths ${ROI_DIR}/aparc+aseg.nii.gz -uthr 2007 -thr 2007 ${ROI_DIR}/rh.fusiform_FS.nii.gz

# left medialorbitofrontal = 1014, right medialorbitofrontal = 2014
fslmaths ${ROI_DIR}/aparc+aseg.nii.gz -uthr 1014 -thr 1014 ${ROI_DIR}/lh.mOFC_FS.nii.gz
fslmaths ${ROI_DIR}/aparc+aseg.nii.gz -uthr 2014 -thr 2014 ${ROI_DIR}/rh.mOFC_FS.nii.gz

# for the lateraloccipital ROIs, subtract V1 and V2. They overlap substantially.
# first binarise the two LO ROIs.
fslmaths ${ROI_DIR}/lh.LOC_FS.nii.gz -bin ${ROI_DIR}/lh.LOC_FS.nii.gz
fslmaths ${ROI_DIR}/rh.LOC_FS.nii.gz -bin ${ROI_DIR}/rh.LOC_FS.nii.gz
#
# now subtract V1 and V2
fslmaths ${ROI_DIR}/lh.LOC_FS.nii.gz -sub ${ROI_DIR}/lh.V1.nii.gz -sub ${ROI_DIR}/lh.V2.nii.gz ${ROI_DIR}/lh.LOC_FS.nii.gz
fslmaths ${ROI_DIR}/rh.LOC_FS.nii.gz -sub ${ROI_DIR}/rh.V1.nii.gz -sub ${ROI_DIR}/rh.V2.nii.gz ${ROI_DIR}/rh.LOC_FS.nii.gz
#
# and binarise again
fslmaths ${ROI_DIR}/lh.LOC_FS.nii.gz -bin ${ROI_DIR}/lh.LOC_FS.nii.gz
fslmaths ${ROI_DIR}/rh.LOC_FS.nii.gz -bin ${ROI_DIR}/rh.LOC_FS.nii.gz

# binarise all ROIs, and merge all lh and rh ROIs
ROIS="V1 V2 hipp_FS ento_FS parahippo_FS LOC_FS fusiform_FS mOFC_FS IT_FS perirhinal"

for ROI in $ROIS
do
  echo $ROI
  fslmaths ${ROI_DIR}/lh.${ROI}.nii.gz -bin ${ROI_DIR}/lh.${ROI}.nii.gz
  fslmaths ${ROI_DIR}/rh.${ROI}.nii.gz -bin ${ROI_DIR}/rh.${ROI}.nii.gz
  fslmaths ${ROI_DIR}/lh.${ROI}.nii.gz -add ${ROI_DIR}/rh.${ROI}.nii.gz ${ROI_DIR}/${ROI}.nii.gz
  fslmaths ${ROI_DIR}/${ROI}.nii.gz -bin ${ROI_DIR}/${ROI}.nii.gz
done

mv ${ROI_DIR}/perirhinal.nii.gz ${ROI_DIR}/PRC_FS.nii.gz