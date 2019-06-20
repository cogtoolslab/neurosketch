#!/bin/bash
#
# prep.sh prepares for analysis of the subject's data
# original author: mason simon (mgsimon@princeton.edu)
# this script was provided by NeuroPipe. modify it to suit your needs

set -e

pushd $(dirname $0) > /dev/null   # move into the subject's directory, quietly
source globals.sh

bash scripts/convert-and-wrap-raw-data.sh $DICOM_ARCHIVE $NIFTI_DIR $SUBJ $RUNORDER_FILE
echo "nifti done -- $(date)"
bash scripts/qa-wrapped-data.sh $NIFTI_DIR $QA_DIR
echo "qa done -- $(date)"
bash scripts/reorient-to-las.sh $NIFTI_DIR
echo "reorient done -- $(date)"

