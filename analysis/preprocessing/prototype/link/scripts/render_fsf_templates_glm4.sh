#!/bin/bash
#
# render-fsf-templates.sh fills in templated fsf files so FEAT can use them
# original author: mason simon (mgsimon@princeton.edu)
# this script was provided by NeuroPipe. modify it to suit your needs
#
# refer to the secondlevel neuropipe tutorial to see an example of how
# to use this script

set -e

source globals.sh

function render_firstlevel {
  fsf_template=$1
  output_dir=$2
  standard_brain=$3
  data_file_prefix=$4
  highres_file=$5
  ev_dir=$6

  subject_dir=$(pwd)

  # note: the following replacements put absolute paths into the fsf file. this
  #       is necessary because FEAT changes directories internally
  cat $fsf_template \
    | sed "s:<?= \$OUTPUT_DIR ?>:$subject_dir/$output_dir:g" \
    | sed "s:<?= \$STANDARD_BRAIN ?>:$standard_brain:g" \
    | sed "s:<?= \$DATA_FILE_PREFIX ?>:$subject_dir/$data_file_prefix:g" \
    | sed "s:<?= \$HIGHRES_FILE ?>:$subject_dir/$highres_file:g" \
    | sed "s:<?= \$EV_DIR ?>:$subject_dir/$ev_dir:g" 
}

runList="1 2 3 4 5 6"
for r in $runList
do

  render_firstlevel $FSF_DIR/glm4.fsf.template \
                    $FIRSTLEVEL_DIR/glm4_recognition_run_${r}.feat \
                    $STANDARD_BRAIN \
                    $NIFTI_DIR/${SUBJ}_recognition_run_${r} \
		            $NIFTI_DIR/${SUBJ}_anat_mprage_brain \
                    $EV_DIR/run_${r} \
                    > $FSF_DIR/glm4_recognition_run_${r}.fsf

done
