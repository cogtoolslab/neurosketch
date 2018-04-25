## Script to grab just FSL GLM analysis output from each subject directory in sketchloop02 project dir on jukebox
## jefan 3/24/17

import os
import shutil
import glob
import sys

# flag = 'all' to get everything inside analysis folder
# 'registration' to get just registration things
flag = 'roimasks'

subjects = ['1121161_neurosketch', '1130161_neurosketch', 
'1201161_neurosketch', '1202161_neurosketch', '1203161_neurosketch',
'1206161_neurosketch', '1206162_neurosketch','1206163_neurosketch',
'1207161_neurosketch','1207162_neurosketch','0110171_neurosketch', '0110172_neurosketch',
'0111171_neurosketch','0112171_neurosketch', '0112172_neurosketch','0112173_neurosketch',
'0113171_neurosketch','0115172_neurosketch','0115174_neurosketch','0117171_neurosketch',
'0118171_neurosketch','0118172_neurosketch','0119171_neurosketch','0119172_neurosketch',
'0119173_neurosketch', '0119174_neurosketch','0120171_neurosketch','0120172_neurosketch',
'0120173_neurosketch','0123171_neurosketch','0123173_neurosketch',
'0124171_neurosketch','0125171_neurosketch','0125172_neurosketch']

print(str(len(subjects)) + ' subjects in total')
## exceptions: 1207161_neurosketch, 0123172_neurosketch 

cwd = os.getcwd()

for s in subjects:
        print s
        if flag == 'all':
                source = '/jukebox/ntb/projects/sketchloop02/subjects/' + s + '/analysis/'                                                                 
                target = s + '/analysis/'

                if os.path.exists(target) is False:
                        os.makedirs(target)
                args = 'rsync -azvh jefan@spock.pni.princeton.edu:' + source + ' ./' + target
                print 'running: ' + args
                os.system(args)


        elif flag == 'registration':
                for i in range(1,7):
                        source = '/jukebox/ntb/projects/sketchloop02/subjects/' + s + '/analysis/firstlevel/reg_recognition_run_' + str(i) + '.feat/reg/'
                        target = s + '/analysis/firstlevel/glm4_recognition_run_' + str(i) + '.feat/reg/'
                        if os.path.exists(target) is False:
                                os.makedirs(target)
                        args = 'rsync -azvh jefan@spock.pni.princeton.edu:' + source + ' ./' + target
                        print 'running: ' + args
                        os.system(args)

	elif flag == 'roimasks':
		source = '/jukebox/ntb/projects/sketchloop02/subjects/' + s + '/analysis/firstlevel/rois/'
		target = s + '/analysis/firstlevel/rois/'
		if os.path.exists(target) is False:
                	os.makedirs(target)
                args = 'rsync -azvh jefan@spock.pni.princeton.edu:' + source + ' ./' + target
                print 'running: ' + args
                os.system(args)
