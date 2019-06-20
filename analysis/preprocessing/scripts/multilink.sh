#!/bin/bash

source globals.sh

origin=$1
subjectList="0110171_neurosketch 0113171_neurosketch 0119172_neurosketch 0123171_neurosketch 1130161_neurosketch 1207162_neurosketch 0110172_neurosketch 0115174_neurosketch 0119173_neurosketch 0123173_neurosketch 1202161_neurosketch 0111171_neurosketch 0117171_neurosketch 0119174_neurosketch 0124171_neurosketch 1203161_neurosketch 0112171_neurosketch 0118171_neurosketch 0120171_neurosketch 0125171_neurosketch 1206161_neurosketch 0112172_neurosketch 0118172_neurosketch 0120172_neurosketch 0125172_neurosketch 1206162_neurosketch 0112173_neurosketch 0119171_neurosketch 0120173_neurosketch 1121161_neurosketch 1206163_neurosketch"

for subject in $subjectList
do
  pushd subjects/${subject}/scripts >/dev/null
  if [ ! -e ${origin} ]
  then
    echo $subject
    #ln -s ../../../prototype/link/scripts/$origin $origin
  fi
  popd >/dev/null
done

