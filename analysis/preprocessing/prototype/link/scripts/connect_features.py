import os
import sys
import numpy as np
import pandas as pd
import nibabel as nib
import itertools

subject = sys.argv[1]
subject = subject.split('_')[0]
proj_dir = sys.argv[2]
dat_type = 'draw'
print(proj_dir)

data_dir = os.path.abspath(os.path.join(proj_dir,'..','..','data'))
feature_dir = os.path.abspath(os.path.join(data_dir, 'features')) 
draw_features = os.path.abspath(os.path.join(feature_dir, 'production')) 
out_dir = os.path.abspath(os.path.join(feature_dir, 'connectivity')) 
if not os.path.isdir(out_dir):
    os.mkdir(out_dir)

roi_list = ['V1Draw', 'V2Draw', 'LOCDraw', 'ParietalDraw']

def load_draw_meta(this_sub):
    this_file = 'metadata_{}_drawing.csv'.format(this_sub)
    x = pd.read_csv(os.path.join(draw_features,this_file))
    return x


def load_draw_feats(this_sub,this_roi):
    this_file = '{}_{}_featurematrix.npy'.format(this_sub,this_roi)
    y = np.load(os.path.join(draw_features,this_file))
    return y


def load_draw_data(this_sub,this_roi):
    x = load_draw_meta(this_sub)
    y = load_draw_feats(this_sub,this_roi)
    assert y.shape[0] == x.shape[0]
    return x,y


all_pairs = itertools.combinations(roi_list, 2)
for (this_roi, that_roi) in all_pairs:
    print(this_roi, that_roi)
    DM_1, DF_1 = load_draw_data(subject, this_roi)
    DM_2, DF_2 = load_draw_data(subject, that_roi)
    assert DF_1.shape[0] == DF_2.shape[0]
    print(DF_1.shape)
    
    this_roi_shape = DF_1.shape[1]
    that_roi_shape = DF_2.shape[1]
    rois_stacked = np.hstack((DF_1, DF_2))
    assert rois_stacked.shape[1] == this_roi_shape + that_roi_shape
    
    trial = 0
    newDF = []
    stackDF = []
    newDM = []
    outmeta = '{}/metadata_{}_corrs.csv'.format(out_dir, subject)

    for ind in range(0, 920, 23):
        if not os.path.exists(outmeta):
            tempDM = np.array(DM_1.iloc[ind].loc[['subj', 'label', 'run_num', 'trial_num']])
            newDM = tempDM if len(newDM) == 0 else np.vstack((newDM, tempDM))
        
        tempDF = rois_stacked[ind:ind+23]
        corrs = np.corrcoef(np.transpose(tempDF))[this_roi_shape:, :this_roi_shape]
        corrs = corrs.flatten()
        newDF = corrs if len(newDF) == 0 else np.vstack((newDF, corrs))
        
        collapse = tempDF.flatten()
        stackDF = collapse if len(stackDF) == 0 else np.vstack((stackDF, collapse))

    if not os.path.exists(outmeta):
        newDM = pd.DataFrame(newDM, columns=['subj', 'label', 'run_num', 'trial_num'])
        newDM.to_csv(outmeta)
    np.save('{}/{}_{}_{}_featurematrix.npy'.format(out_dir, subject, this_roi, that_roi), newDF)
    np.save('{}/{}_{}_{}_stackmatrix.npy'.format(out_dir, subject, this_roi, that_roi), stackDF)