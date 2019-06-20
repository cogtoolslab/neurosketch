#!/bin/bash

set -e 
source globals.sh


# align mask to this subject's hi-resolution image

FS_DIR=$FIRSTLEVEL_DIR/surfROI
OUT_DIR=$FIRSTLEVEL_DIR/rois/drawROI
PARC=$FIRSTLEVEL_DIR/surfROI/aparc+aseg.nii.gz

roi_list="draw_task_mask"

for roi in $roi_list
do

    INPUT=$OUT_DIR/${roi}_HR.nii.gz
    OUTPUT=$OUT_DIR/${roi}_HR_bin.nii.gz
    CROSS=$OUT_DIR/${roi}_surf.nii.gz
    fslmaths $INPUT -thr 0.5 -bin $OUTPUT
    fslmaths $OUTPUT -mul $PARC $CROSS
    echo "${roi} binarized -- $(date)"
    wait

done


# precentral (lh - 1024, rh - 2024)
fslmaths $CROSS -uthr 1024 -thr 1024 -bin ${OUT_DIR}/lh.preCentral.nii.gz
fslmaths $CROSS -uthr 2024 -thr 2024 -bin ${OUT_DIR}/rh.preCentral.nii.gz
fslmaths ${OUT_DIR}/lh.preCentral.nii.gz -add ${OUT_DIR}/rh.preCentral.nii.gz ${OUT_DIR}/preCentral_draw.nii.gz
fslmaths ${OUT_DIR}/preCentral_draw.nii.gz -bin ${OUT_DIR}/preCentral_draw.nii.gz

# frontal (lh - 1003, 1018-1020, 1027-1028, rh - 2003, 2018-2020, 2027-2028)
fslmaths $CROSS -uthr 1003 -thr 1003 -bin ${OUT_DIR}/lh.Frontal1.nii.gz
fslmaths $CROSS -uthr 2003 -thr 2003 -bin ${OUT_DIR}/rh.Frontal1.nii.gz
fslmaths $CROSS -uthr 1018 -thr 1020 -bin ${OUT_DIR}/lh.Frontal2.nii.gz
fslmaths $CROSS -uthr 2018 -thr 2020 -bin ${OUT_DIR}/rh.Frontal2.nii.gz
fslmaths $CROSS -uthr 1027 -thr 1028 -bin ${OUT_DIR}/lh.Frontal3.nii.gz
fslmaths $CROSS -uthr 2027 -thr 2028 -bin ${OUT_DIR}/rh.Frontal3.nii.gz
fslmaths ${OUT_DIR}/lh.Frontal1.nii.gz -add ${OUT_DIR}/rh.Frontal1.nii.gz -add ${OUT_DIR}/lh.Frontal2.nii.gz -add ${OUT_DIR}/rh.Frontal2.nii.gz -add ${OUT_DIR}/lh.Frontal3.nii.gz -add ${OUT_DIR}/rh.Frontal3.nii.gz ${OUT_DIR}/Frontal_draw.nii.gz
fslmaths ${OUT_DIR}/Frontal_draw.nii.gz -bin ${OUT_DIR}/Frontal_draw.nii.gz

# supramarginal (lh - 1031, rh - 2031)
fslmaths $CROSS -uthr 1031 -thr 1031 -bin ${OUT_DIR}/lh.supraMarginal.nii.gz
fslmaths $CROSS -uthr 2031 -thr 2031 -bin ${OUT_DIR}/rh.supraMarginal.nii.gz
fslmaths ${OUT_DIR}/lh.supraMarginal.nii.gz -add ${OUT_DIR}/rh.supraMarginal.nii.gz ${OUT_DIR}/supraMarginal_draw.nii.gz
fslmaths ${OUT_DIR}/supraMarginal_draw.nii.gz -bin ${OUT_DIR}/supraMarginal_draw.nii.gz

# parietal (lh - 1008, 1029, rh - 2008, 2029)
fslmaths $CROSS -uthr 1008 -thr 1008 -bin ${OUT_DIR}/lh.Parietal1.nii.gz
fslmaths $CROSS -uthr 2008 -thr 2008 -bin ${OUT_DIR}/rh.Parietal1.nii.gz
fslmaths $CROSS -uthr 1029 -thr 1029 -bin ${OUT_DIR}/lh.Parietal2.nii.gz
fslmaths $CROSS -uthr 2029 -thr 2029 -bin ${OUT_DIR}/rh.Parietal2.nii.gz
fslmaths ${OUT_DIR}/lh.Parietal1.nii.gz -add ${OUT_DIR}/rh.Parietal1.nii.gz -add ${OUT_DIR}/lh.Parietal2.nii.gz -add ${OUT_DIR}/rh.Parietal2.nii.gz ${OUT_DIR}/Parietal_draw.nii.gz
fslmaths ${OUT_DIR}/Parietal_draw.nii.gz -bin ${OUT_DIR}/Parietal_draw.nii.gz

# postcentral (lh - 1022, rh - 2022)
fslmaths $CROSS -uthr 1022 -thr 1022 -bin ${OUT_DIR}/lh.postCentral.nii.gz
fslmaths $CROSS -uthr 2022 -thr 2022 -bin ${OUT_DIR}/rh.postCentral.nii.gz
fslmaths ${OUT_DIR}/lh.postCentral.nii.gz -add ${OUT_DIR}/rh.postCentral.nii.gz ${OUT_DIR}/postCentral_draw.nii.gz
fslmaths ${OUT_DIR}/postCentral_draw.nii.gz -bin ${OUT_DIR}/postCentral_draw.nii.gz

# postcentral (lh - 1035, rh - 2035)
fslmaths $CROSS -uthr 1035 -thr 1035 -bin ${OUT_DIR}/lh.Insula.nii.gz
fslmaths $CROSS -uthr 2035 -thr 2035 -bin ${OUT_DIR}/rh.Insula.nii.gz
fslmaths ${OUT_DIR}/lh.Insula.nii.gz -add ${OUT_DIR}/rh.Insula.nii.gz ${OUT_DIR}/Insula_draw.nii.gz
fslmaths ${OUT_DIR}/Insula_draw.nii.gz -bin ${OUT_DIR}/Insula_draw.nii.gz


#V1 V2 LOC_FS
fslmaths $OUTPUT -mul $FS_DIR/V1.nii.gz -bin ${OUT_DIR}/V1_draw.nii.gz
fslmaths $OUTPUT -mul $FS_DIR/V2.nii.gz -bin ${OUT_DIR}/V2_draw.nii.gz
fslmaths $OUTPUT -mul $FS_DIR/LOC_FS.nii.gz -bin ${OUT_DIR}/LOC_draw.nii.gz

rm ${OUT_DIR}/lh*
rm ${OUT_DIR}/rh*