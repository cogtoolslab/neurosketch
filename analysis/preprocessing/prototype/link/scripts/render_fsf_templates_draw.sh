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


ev_files=$(ls $EV_DIR2)
set -- $ev_files
ev_file1=$1
ev_file2=$2


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
    | sed "s:<?= \$EV_FILE1 ?>:$subject_dir/$ev_dir/$ev_file1:g" \
    | sed "s:<?= \$EV_FILE2 ?>:$subject_dir/$ev_dir/$ev_file2:g" 
}


runList="1 2 3 4"
for r in $runList
do
  render_firstlevel $FSF_DIR/draw.fsf.template \
                    $FIRSTLEVEL_DIR/draw_run_${r}.feat \
                    $STANDARD_BRAIN \
                    $NIFTI_DIR/${SUBJ}_drawing_run_${r} \
                    $NIFTI_DIR/${SUBJ}_anat_mprage_brain \
                    $EV_DIR2 \
		            > $FSF_DIR/draw_run_${r}.fsf
done
