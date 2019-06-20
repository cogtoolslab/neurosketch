#!/bin/bash
# author: mgsimon@princeton.edu

subject_dir=$1

SLEEP_INTERVAL=10   # this is in seconds

while [ ! -e "${subject_dir}/analysis/firstlevel/freesurfer/surfdone" ]; do
  # freesurfer is still running...
  sleep $SLEEP_INTERVAL
done

