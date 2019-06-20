#!/bin/bash

set -e 

CURRDIR=$(pwd)
OUTDIR=$CURRDIR

source globals.sh
subj_mask_dir=$SUBJECT_DIR/"analysis/firstlevel/rois"

echo "Now binarizing mask..."

runList="1 2 3 4 5 6"
for r in $runList
do

	#inferior temporal (ant, pos, occ)
	echo "IT"
	INPUT=$subj_mask_dir/"IT_func_run_${r}.nii.gz"
	OUTPUT=$subj_mask_dir/"IT_func_run_${r}_binarized.nii.gz"
	fslmaths $INPUT -thr 0.9 -bin $OUTPUT
	wait

	#parahippocampal (ant, pos)
	echo "parahippocampal"
	INPUT=$subj_mask_dir/"parahippo_func_run_${r}.nii.gz"
	OUTPUT=$subj_mask_dir/"parahippo_func_run_${r}_binarized.nii.gz"
	fslmaths $INPUT -thr 0.9 -bin $OUTPUT
	wait

	#fusiform (ant, pos, occ)
	echo "fusiform"
	INPUT=$subj_mask_dir/"fusiform_func_run_${r}.nii.gz"
	OUTPUT=$subj_mask_dir/"fusiform_func_run_${r}_binarized.nii.gz"
	fslmaths $INPUT -thr 0.9 -bin $OUTPUT
	wait

	# LOC
	echo "LOC"
	INPUT=$subj_mask_dir/"LOC_func_run_${r}.nii.gz"
	OUTPUT=$subj_mask_dir/"LOC_func_run_${r}_binarized.nii.gz"
	fslmaths $INPUT -thr 0.9 -bin $OUTPUT

	# V1
	echo "V1"
	INPUT=$subj_mask_dir/"V1_func_run_${r}.nii.gz"
	OUTPUT=$subj_mask_dir/"V1_func_run_${r}_binarized.nii.gz"
	fslmaths $INPUT -thr 0.9 -bin $OUTPUT

	# occitemp: V1, LOC, fusiform, parahippocampal, IT
	echo "occitemp"
	INPUT=$subj_mask_dir/"occitemp_func_run_${r}.nii.gz"
	OUTPUT=$subj_mask_dir/"occitemp_func_run_${r}_binarized.nii.gz"
	fslmaths $INPUT -thr 0.9 -bin $OUTPUT
	


done