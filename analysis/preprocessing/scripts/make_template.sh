
#!/bin/bash
#
# render-fsf-templates.sh fills in templated fsf files so FEAT can use them
# original author: mason simon (mgsimon@princeton.edu)
# this script was provided by NeuroPipe. modify it to suit your needs
#
# refer to the secondlevel neuropipe tutorial to see an example of how
# to use this script

#set -e

source globals.sh

NOW="${PROJECT_DIR}"
othersubjects=$1
SUBJ=$2
NIFTI_DIR="${NOW}/subjects/${SUBJ}/data/nifti"
SECONDLEVEL_DIR="${NOW}/subjects/${SUBJ}/analysis/secondlevel"
STANDARD_BRAIN="${NOW}/standard/MNI152_T1_2mm_brain"




function render_secondlevel {
  fsf_template=$1
  project_dir=$2
  output_dir=$3
  standard_brain=$4
  highres_file=$5
  subject=$6
  otherSubjects=$7

  subject_dir="subjects/${subject}"
  
  fdavw=""
  evv=""
  groupmem=""
  runnum=0
  for subject in $otherSubjects
  do
    ((runnum++))
    _fdavw="# 4D AVW data or FEAT directory ($runnum)\nset feat_files(${runnum}) "${project_dir}/subjects/${subject}/analysis/secondlevel/draw.gfeat/cope3.feat"\n\n"
    _evv="# Higher-level EV value for EV 1 and input $runnum\nset fmri(evg${runnum}.1) 1\n\n"
    _groupmem="# Group membership for input $runnum\nset fmri(groupmem.${runnum}) 1\n\n"
    fdavw=$fdavw$_fdavw
    evv=$evv$_evv
    groupmem=$groupmem$_groupmem
  done

  # note: the following replacements put absolute paths into the fsf file. this
  #       is necessary because FEAT changes directories internally
  cat $fsf_template \
    | sed "s:<?= \$OUTPUT_DIR ?>:$output_dir:g" \
    | sed "s:<?= \$STANDARD_BRAIN ?>:$standard_brain:g" \
    | sed "s:<?= \$HIGHRES_FILE ?>:$highres_file:g" \
    | sed "s:<?= \$4D_AVW ?>:$fdavw:g" \
    | sed "s:<?= \$EV_VALUE ?>:$evv:g" \
    | sed "s:<?= \$GROUP_MEM ?>:$groupmem:g" \
    | sed "s:<?= \$NRUNS ?>:$runnum:g"
}


render_secondlevel "${NOW}/scripts/populate.fsf.template" \
                   "${NOW}" \
                   "${NOW}/group/${SUBJ}.gfeat" \
                   "${STANDARD_BRAIN}" \
                   "${NIFTI_DIR}/${SUBJ}_anat_mprage_brain.nii.gz" \
                   "$SUBJ" \
                   "$othersubjects" \
                   > "${NOW}/group/${SUBJ}.fsf"
