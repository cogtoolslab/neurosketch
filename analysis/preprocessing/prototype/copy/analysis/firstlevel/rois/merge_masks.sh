#!/bin/bash

set -e 

CURRDIR=$(pwd)
OUTDIR=$CURRDIR

source globals.sh
subj_mask_dir=$SUBJECT_DIR/"analysis/firstlevel/rois"

echo "Now merging pairs of masks..."

runList="1 3 5"
for r in $runList
do
	# get the index of the second run in the pair
	PAIR_IND="$(($r + 1))"

	#inferior temporal (ant, pos, occ)
	echo "IT"
	INPUT1=$subj_mask_dir/"IT_func_run_${r}_binarized.nii.gz"
	INPUT2=$subj_mask_dir/"IT_func_run_${PAIR_IND}_binarized.nii.gz"
	OUTPUT=$subj_mask_dir/"IT_func_combined_${r}${PAIR_IND}_binarized.nii.gz"
	fslmaths $INPUT1 -add $INPUT2 -thr 1.5 -bin $OUTPUT
	wait

	#parahippocampal (ant, pos)
	echo "parahippocampal"
	INPUT1=$subj_mask_dir/"parahippo_func_run_${r}_binarized.nii.gz"
	INPUT2=$subj_mask_dir/"parahippo_func_run_${PAIR_IND}_binarized.nii.gz"
	OUTPUT=$subj_mask_dir/"parahippo_func_combined_${r}${PAIR_IND}_binarized.nii.gz"
	fslmaths $INPUT1 -add $INPUT2 -thr 1.5 -bin $OUTPUT
	wait

	#fusiform (ant, pos, occ)
	echo "fusiform"
	INPUT1=$subj_mask_dir/"fusiform_func_run_${r}_binarized.nii.gz"
	INPUT2=$subj_mask_dir/"fusiform_func_run_${PAIR_IND}_binarized.nii.gz"
	OUTPUT=$subj_mask_dir/"fusiform_func_combined_${r}${PAIR_IND}_binarized.nii.gz"
	fslmaths $INPUT1 -add $INPUT2 -thr 1.5 -bin $OUTPUT
	wait

	# LOC
	echo "LOC"
	INPUT1=$subj_mask_dir/"LOC_func_run_${r}_binarized.nii.gz"
	INPUT2=$subj_mask_dir/"LOC_func_run_${PAIR_IND}_binarized.nii.gz"
	OUTPUT=$subj_mask_dir/"LOC_func_combined_${r}${PAIR_IND}_binarized.nii.gz"
	fslmaths $INPUT1 -add $INPUT2 -thr 1.5 -bin $OUTPUT
	wait

	# V1
	echo "V1"
	INPUT1=$subj_mask_dir/"V1_func_run_${r}_binarized.nii.gz"
	INPUT2=$subj_mask_dir/"V1_func_run_${PAIR_IND}_binarized.nii.gz"
	OUTPUT=$subj_mask_dir/"V1_func_combined_${r}${PAIR_IND}_binarized.nii.gz"
	fslmaths $INPUT1 -add $INPUT2 -thr 1.5 -bin $OUTPUT

	# occitemp
	echo "occitemp"
	INPUT1=$subj_mask_dir/"occitemp_func_run_${r}_binarized.nii.gz"
	INPUT2=$subj_mask_dir/"occitemp_func_run_${PAIR_IND}_binarized.nii.gz"
	OUTPUT=$subj_mask_dir/"occitemp_func_combined_${r}${PAIR_IND}_binarized.nii.gz"
	fslmaths $INPUT1 -add $INPUT2 -thr 1.5 -bin $OUTPUT	

done