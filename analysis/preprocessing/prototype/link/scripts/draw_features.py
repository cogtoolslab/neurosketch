import os
import sys
import numpy as np
import pandas as pd
import nibabel as nib

subject = sys.argv[1]
subject = subject.split('_')[0]
proj_dir = sys.argv[2]
dat_type = 'draw'
print(proj_dir)

data_dir = os.path.abspath(os.path.join(proj_dir,'..','..','data'))
feature_dir = os.path.abspath(os.path.join(data_dir, 'features')) 
prod_regdir = os.path.abspath(os.path.join(proj_dir,'subjects/{}_neurosketch/draw_reg'))
prod_reg = os.path.abspath(os.path.join(prod_regdir,'{}.txt'))
filt_func = os.path.abspath(os.path.join(proj_dir,'subjects/{}_neurosketch/analysis/firstlevel/parameter',
                                         '{}_neurosketch_{}_run_{}_filtfuncHIRES.nii.gz'))
roi_dir = os.path.abspath(os.path.join(proj_dir,'subjects/{}_neurosketch/analysis/firstlevel/surfROI'))
out_dir = os.path.abspath(os.path.join(feature_dir, 'production')) 
if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

objects = [txt.split('.txt')[0] for txt in os.listdir(prod_regdir.format(subject))]
print(objects)
roi_list_masks = ['V1','V2','LOC_FS','IT_FS','fusiform_FS','parahippo_FS','PRC_FS','ento_FS','hipp_FS',
                  'V1Draw', 'V2Draw', 'LOCDraw', 'ParietalDraw']
roi_list_names = ['V1','V2','LOC','IT','fusiform','parahippo','PRC','ento','hipp', 
                  'V1Draw', 'V2Draw', 'LOCDraw', 'ParietalDraw']


for phase in ['1234']:
    # initialize data columns
    subj = [subject] * 920
    label = []
    run_num = [phase[0]]*230 + [phase[1]]*230 + [phase[2]]*230 + [phase[3]]*230
    TR_num = []

    for rn, run in enumerate(phase):
        # load subject's time series for this run
        timeseries = nib.load(filt_func.format(subject, subject, dat_type, run))
        timeseries = timeseries.get_data().transpose((3, 0, 1, 2))

        # use information in regressor/run_x folder to make hasImage vector
        # associated TR is just the hasImage index, converted to a float
        Onsets = [0]*308
        for obj in objects:
            with open(prod_reg.format(subject, obj)) as f:
                times = [line.split(' ')[0] for line in f.read().split('\n')[:-1]]
                for t in times:
                    TR = int(float(t)/1.5)
                    print(t, TR, TR+23)
                    for tr in range(TR, TR+23):
                        Onsets[tr] = obj

        # wherever hasImage, we want the features
        features = [timeseries[n+3] for n, onset in enumerate(Onsets) if onset != 0]
        labels = [label for label in Onsets if label != 0]
        FEATURES = np.array(features) if rn == 0 else np.vstack((FEATURES, np.array(features)))
        LABELS = labels if rn == 0 else LABELS + labels
    np.save('{}/{}_featurematrix.npy'.format(out_dir, subject), FEATURES)
    
    for roi, roiname in zip(roi_list_masks[:], roi_list_names[:]):
        mask = nib.load('{}/{}.nii.gz'.format(roi_dir.format(subject), roi))
        maskDat = mask.get_data()
        masked = FEATURES[:, maskDat == 1]
        np.save('{}/{}_{}_featurematrix.npy'.format(out_dir, subject, roiname), masked)
        
        ## metadata
        x = pd.DataFrame([subj, LABELS, run_num, TR_num]) # lists of the same length
        x = x.transpose()
        x.columns = ['subj','label','run_num', 'TR_num']
        x.to_csv('{}/metadata_{}_drawing.csv'.format(out_dir, subject))
