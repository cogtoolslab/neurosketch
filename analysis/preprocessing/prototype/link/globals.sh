#!/bin/bash
# author: mgsimon@princeton.edu
# this script sets up global variables for the analysis of the current subject

set -e # stop immediately when an error occurs

# add necessary directories to the system path
export BXH_DIR=/jukebox/ntb/packages/bxh_xcede_tools/bin
export MAGICK_HOME=/jukebox/ntb/packages/ImageMagick-6.5.9-9
export BIAC_HOME=/jukebox/ntb/packages/BIAC_matlab/mr

source scripts/subject_id.sh  # this loads the variable SUBJ
_PROJ_DIR=$(pwd)
PROJ_DIR="${_PROJ_DIR}/../.."
SUBJECT_DIR=$PROJ_DIR/subjects/$SUBJ
SPINECHO_DIR=$PROJ_DIR/spin_echo_params

RUNORDER_FILE=run-order.txt

DATA_DIR=data
SCRIPT_DIR=scripts
FSF_DIR=fsf
DICOM_ARCHIVE=data/raw.tar.gz
NIFTI_DIR=data/nifti
QA_DIR=data/qa
BEHAVIORAL_DATA_DIR=data/behavioral
FIRSTLEVEL_DIR=analysis/firstlevel
SECONDLEVEL_DIR=analysis/secondlevel
OUTPUT_DIR=$FIRSTLEVEL_DIR/parameter
EV_DIR=regressor
EV_DIR2=draw_reg
BEHAVIORAL_OUTPUT_DIR=output/behavioral
STANDARD_BRAIN=$PROJ_DIR/standard/MNI152_T1_2mm_brain
CONDA_ENV=prep
