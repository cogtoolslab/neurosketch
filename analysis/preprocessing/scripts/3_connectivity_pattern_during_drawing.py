import os
import sys
import numpy as np
import warnings
import itertools
import pandas as pd
warnings.filterwarnings("ignore")




curr_dir = os.getcwd()
roi_list_connect = np.array(['V1Draw', 'V2Draw', 'LOCDraw', 'ParietalDraw', 'preCentralDraw'])
# root paths
proj_dir = os.path.abspath(os.path.join(curr_dir,'..','..'))  # use relative paths
print(proj_dir)
data_dir = os.path.abspath(os.path.join(curr_dir,'..','..','data'))  # use relative paths 'D:\\data'
path_to_connect = os.path.join(data_dir, 'features/connectivity')
results_dir = os.path.join(proj_dir, 'results')
nb_name = '3_connectivity_pattern_during_drawing'

# add helpers to python path
import sys
if os.path.join(proj_dir, 'analysis', 'utils') not in sys.path:
    sys.path.append(os.path.join(proj_dir, 'analysis', 'utils'))

# module definitions
import object_evidence_analysis_helpers as utils
utils.data_dir = data_dir
utils.path_to_connect = path_to_connect
utils.roi_list_connect = roi_list_connect

# root paths
proj_dir = os.path.abspath(os.path.join(curr_dir,'..','..'))  # use relative paths
data_dir = os.path.abspath(os.path.join(curr_dir,'..','..','data'))  # use relative paths 'D:\\data'
path_to_connect = os.path.join(data_dir, 'features/connectivity')
results_dir = os.path.join(proj_dir, 'results')
csv_dir = os.path.join(proj_dir, 'results', 'csv')
nb_name = '3_connectivity_pattern_during_drawing'

utils.data_dir = data_dir
utils.path_to_connect = path_to_connect
utils.roi_list_connect = roi_list_connect

## get raw file list for connectivity features
CONNECT_METAS = sorted([i for i in os.listdir(path_to_connect) if (i.split('.')[-1]=='csv')])
CONNECT_FEATS = sorted([i for i in os.listdir(path_to_connect) if (i.split('.')[-1]=='npy')])
CONNECT_SUBS = np.array([i.split('_')[0] for i in CONNECT_FEATS])

sub_list = np.unique(CONNECT_SUBS)
print('We have data from {} subjects.'.format(len(sub_list)))


version = 'phase'  # 'phase', 'allruns'
logged = True
feattype = sys.argv[1]



ALLDM, Acc = utils.make_drawing_connectivity_predictions(sub_list[:],roi_list_connect,
                                                         version=version, logged=logged, feature_type=feattype)
# save out ALLDM & Acc
Acc = np.array(Acc)
np.save(os.path.join(csv_dir,
                     '{}_{}_accuracy_production.npy'.format(feattype, version)), Acc)
ALLDM.to_csv(os.path.join(csv_dir,
                          '{}_{}_logprobs_production.csv'.format(feattype, version)), index=False)

ALLDM['phase_num'] = np.where(ALLDM['run_num']>3, 1, 2)
ALLDM.to_csv(os.path.join(csv_dir,
                          '{}_{}_logprobs_production.csv'.format(feattype, version)),index=False)
print('Done!')
