#!/bin/bash

set -e 

CURRDIR=$(pwd)
OUTDIR=$CURRDIR

#inferior temporal (ant, pos, occ)
echo "IT"
INPUT=$CURRDIR/"IT_hires.nii.gz"
OUTPUT=$OUTDIR/"IT_hires_binarized.nii.gz"
fslmaths $INPUT -thr 0.9 -bin $OUTPUT
wait

#parahippocampal (ant, pos)
echo "parahippocampal"
INPUT=$CURRDIR/"parahippo_hires.nii.gz"
OUTPUT=$OUTDIR/"parahippo_hires_binarized.nii.gz"
fslmaths $INPUT -thr 0.9 -bin $OUTPUT
wait

#fusiform (ant, pos, occ)
echo "fusiform"
INPUT=$CURRDIR/"fusiform_hires.nii.gz"
OUTPUT=$OUTDIR/"fusiform_hires_binarized.nii.gz"
fslmaths $INPUT -thr 0.9 -bin $OUTPUT
wait

# LOC
echo "LOC"
INPUT=$CURRDIR/"LOC_hires.nii.gz"
OUTPUT=$OUTDIR/"LOC_hires_binarized.nii.gz"
fslmaths $INPUT -thr 0.9 -bin $OUTPUT

# V1
echo "V1"
INPUT=$CURRDIR/"V1_hires.nii.gz"
OUTPUT=$OUTDIR/"V1_hires_binarized.nii.gz"
fslmaths $INPUT -thr 0.9 -bin $OUTPUT
