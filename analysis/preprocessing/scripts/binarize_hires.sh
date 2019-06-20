#!/bin/bash

set -e



# align mask to this subject's hi-resolution image

SUBJ=$1
FIRSTLEVEL_DIR=subjects/${SUBJ}/analysis/firstlevel
NIFTI_DIR=subjects/${SUBJ}/data/nifti
OUT_DIR=$FIRSTLEVEL_DIR/surfROI
FEAT_DIR=group/${SUBJ}.gfeat
PARC=$FIRSTLEVEL_DIR/surfROI/aparc+aseg.nii.gz
HIRES=$NIFTI_DIR/${SUBJ}_anat_mprage_brain.nii.gz
STD2HIRES=$FIRSTLEVEL_DIR/draw_run_1.feat/reg/standard2highres.mat

roi="draw_task_mask"

# CONVERT DRAW TASK MASK FROM STANDARD TO SUBJECT ANATOMICAL SPACE
INPUT=$FEAT_DIR/${roi}.nii.gz
OUTPUT=$OUT_DIR/${roi}_HR.nii.gz
flirt -ref $HIRES -in $INPUT -out $OUTPUT -applyisoxfm 2 -init $STD2HIRES
echo "${roi} done -- $(date)"

# BINARIZE THIS NEWLY TRANSFORMED MASK AND CROSS IT WITH FREESURFER SEGMENTATION
INPUT=$OUT_DIR/${roi}_HR.nii.gz
OUTPUT=$OUT_DIR/${roi}_HR_bin.nii.gz
CROSS=$OUT_DIR/${roi}_surf.nii.gz
fslmaths $INPUT -thr 3 -bin $OUTPUT
fslmaths $OUTPUT -mul $PARC $CROSS
echo "${roi} binarized -- $(date)"
wait

# parietal (lh - 1008, 1029, rh - 2008, 2029)
fslmaths $CROSS -uthr 1008 -thr 1008 -bin ${OUT_DIR}/lh.Parietal1.nii.gz
fslmaths $CROSS -uthr 2008 -thr 2008 -bin ${OUT_DIR}/rh.Parietal1.nii.gz
fslmaths $CROSS -uthr 1029 -thr 1029 -bin ${OUT_DIR}/lh.Parietal2.nii.gz
fslmaths $CROSS -uthr 2029 -thr 2029 -bin ${OUT_DIR}/rh.Parietal2.nii.gz
fslmaths ${OUT_DIR}/lh.Parietal1.nii.gz -add ${OUT_DIR}/rh.Parietal1.nii.gz -add ${OUT_DIR}/lh.Parietal2.nii.gz -add ${OUT_DIR}/rh.Parietal2.nii.gz ${OUT_DIR}/ParietalDraw.nii.gz
fslmaths ${OUT_DIR}/ParietalDraw.nii.gz -bin ${OUT_DIR}/ParietalDraw.nii.gz

#V1 V2 LOC_FS
fslmaths $OUTPUT -mul $OUT_DIR/V1.nii.gz -bin ${OUT_DIR}/V1Draw.nii.gz
fslmaths $OUTPUT -mul $OUT_DIR/V2.nii.gz -bin ${OUT_DIR}/V2Draw.nii.gz
fslmaths $OUTPUT -mul $OUT_DIR/LOC_FS.nii.gz -bin ${OUT_DIR}/LOCDraw.nii.gz

rm ${OUT_DIR}/lh*
rm ${OUT_DIR}/rh*
