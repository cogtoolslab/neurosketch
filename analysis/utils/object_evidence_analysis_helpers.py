from __future__ import division
import os
import pandas as pd
import numpy as np
from numpy import shape
import sklearn
from sklearn import linear_model
from nilearn import image
from sklearn.metrics.pairwise import pairwise_distances
import nibabel as nib

from scipy.stats import norm, linregress
import scipy.stats as stats
import itertools
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import rcParams
rcParams['pdf.fonttype'] = 42
rcParams.update({'figure.autolayout': True})   
import seaborn as sns
from IPython.display import clear_output
sns.set_context('talk')
colors = sns.color_palette("cubehelix", 5)


################### GLOBALS ###################################################################

curr_dir = os.getcwd()
proj_dir = os.path.abspath(os.path.join(curr_dir,'..','..')) ## use relative paths
data_dir = os.path.abspath(os.path.join(curr_dir,'..','..','data')) ## use relative paths 'D:\\data'
results_dir = os.path.join(proj_dir, 'results','csv')
plot_dir = os.path.join(proj_dir,'results','plots')

################### GENERAL HELPERS ###########################################################

def load_draw_meta(this_sub):
    this_file = 'metadata_{}_drawing.csv'.format(this_sub)
    x = pd.read_csv(os.path.join(path_to_draw,this_file))
    x = x.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1)
    x['trial_num'] = np.repeat(np.arange(40),23)
    return x

def load_draw_feats(this_sub,this_roi):
    this_file = '{}_{}_featurematrix.npy'.format(this_sub,this_roi)
    y = np.load(os.path.join(path_to_draw,this_file))
    y = y.transpose()
    return y

def load_draw_data(this_sub,this_roi):
    x = load_draw_meta(this_sub)
    y = load_draw_feats(this_sub,this_roi)
    assert y.shape[0] == x.shape[0]
    return x,y

def load_recog_meta(this_sub,this_roi,this_phase):
    this_file = 'metadata_{}_{}_{}.csv'.format(this_sub,this_roi,this_phase)
    x = pd.read_csv(os.path.join(path_to_recog,this_file))
    x = x.drop(['Unnamed: 0'], axis=1)
    return x

def load_recog_feats(this_sub,this_roi,this_phase):
    this_file = '{}_{}_{}_featurematrix.npy'.format(this_sub,this_roi,this_phase)
    y = np.load(os.path.join(path_to_recog,this_file))
    y = y.transpose()
    return y

def load_recog_data(this_sub,this_roi,this_phase):
    x = load_recog_meta(this_sub,this_roi,this_phase)
    y = load_recog_feats(this_sub,this_roi,this_phase)
    assert y.shape[0] == x.shape[0]
    return x,y

def load_connect_meta(this_sub):
    this_file = 'metadata_{}_corrs.csv'.format(this_sub)
    x = pd.read_csv(os.path.join(path_to_connect,this_file))
    return x

def load_connect_feats(this_sub,this_roi_pair, feature_type='connect'):
    suffix = 'feature' if feature_type == 'connect' else 'stack'
    this_file = '{}_{}_{}matrix.npy'.format(this_sub,this_roi_pair, suffix)
    y = np.load(os.path.join(path_to_connect, this_file))
    return y

def load_connect_data(this_sub,this_roi_pair, feature_type='connect'):
    x = load_connect_meta(this_sub)
    y = load_connect_feats(this_sub,this_roi_pair, feature_type)
    assert y.shape[0] == x.shape[0]
    return x, y

def normalize(X):
    '''
    z-score normalization to de-mean & standardize variances within-voxel
    '''
    X = X - X.mean(0)
    X = X / np.maximum(X.std(0), 1e-5)
    return X

def flatten(x):
    return [item for sublist in x for item in sublist]

def cleanup_df(df):
    surplus = [i for i in df.columns if 'Unnamed' in i]
    df = df.drop(surplus,axis=1)
    return df    

def bootstrapCI(x,nIter=1000,crit_val = 0):
    '''
    input: x is an array
    '''
    u = []
    for i in np.arange(nIter):
        inds = np.random.RandomState(i).choice(len(x),len(x))
        boot = x[inds]
        u.append(np.mean(boot))

    p1 = len([i for i in u if i<crit_val])/len(u) * 2 ## first version of p-value reflects number of samples that have value below crit_val
    p2 = len([i for i in u if i>crit_val])/len(u) * 2 ## second version of p-value reflects number of samples that have value above crit_val
    U = np.mean(u)
    lb = np.percentile(u,2.5)
    ub = np.percentile(u,97.5)
    return U,lb,ub,p1,p2

###############################################################################################
################### SPECIFIC HELPERS FOR JUPYTER NOTEBOOKS ####################################
###############################################################################################


def make_drawing_predictions(sub_list,roi_list,version='4way',logged=True,C=1):
    '''
    input:
        sub_list: a list containing subject IDs
        roi_list: a list containing roi names
        version: a string from options: ['4way','3way','2way']
            4way: trains to discriminate all four objects from recognition runs
            4wayIndependent: subsamples one of the trained objects, trains
                3way classifier that outputs probabilities for the subsampled trained 
                and all control objects; control probabilities are aggregated across
                classifiers while trained probabilities aren't, resulting in 4 scores per row
            3way: subsamples one of the control objects, trains 3-way classifier
                    that outputs probabilities for target, foil, and control objects
                    that is then aggregated across classifiers
            2way: trains to discriminate only the two trained objects from recognition runs
                    then makes predictions on drawing data
            2wayDraw: trains to discriminate only the two trained objects on three drawing runs
                      and makes predictions on the held out drawing run, for all runs
        logged: boolean. If true, return log-probabilities. If false, return raw probabilities.

    assumes: that you have directories containing recognition run and drawing run data, consisting of paired .npy
                voxel matrices and .csv metadata matrices
    '''

    ## Handle slightly different naming for same ROIs in the drawing/recog data directories
    # ROI labels in the drawing data directory
    roi_list_draw = np.array(['V1Draw', 'V2Draw', 'LOCDraw', 'parietalDraw', 
                         'smgDraw', 'sensoryDraw', 'motorDraw', 'frontalDraw'])
    # ROI labels in the recog data directory
    roi_list_recog = np.array(['V1Draw', 'V2Draw', 'LOCDraw', 'ParietalDraw', 
                         'supraMarginalDraw', 'postCentralDraw', 'preCentralDraw', 'FrontalDraw'])
    # bidirectional dictionaries to map from one to the other
    draw_to_recog_roi_dict = dict(zip(roi_list_draw,roi_list_recog))
    recog_to_draw_roi_dict = dict(zip(roi_list_recog,roi_list_draw))    
    
    # initialize "All Data Matrix"
    ALLDM = []
    ## loop through all subjects and rois
    Acc = []
    for this_roi in roi_list:
        acc = []
        for this_sub in sub_list:
            print('Now analyzing ROI: {} from subject: {} ...'.format(this_roi,this_sub))
            clear_output(wait=True)            
            ## load subject data in
            DM, DF = load_draw_data(this_sub,this_roi)
            try:
                RM, RF = load_recog_data(this_sub,this_roi,'12')
            except:
                that_roi = draw_to_recog_roi_dict[this_roi]
                RM, RF = load_recog_data(this_sub,that_roi,'12')
            
            assert RF.shape[1]==DF.shape[1] ## that number of voxels is identical

            # identify control objects;
            # we wil train one classifier with
            trained_objs = sorted(np.unique(DM.label.values))
            control_objs = sorted([i for i in ['bed','bench','chair','table'] if i not in trained_objs])
            probs = []
            logprobs = []

            if version=='4way':
                ## normalize voxels within task
                normalize_on = 1
                if normalize_on:
                    _RF = normalize(RF)
                    _DF = normalize(DF)
                else:
                    _RF = RF
                    _DF = DF

                # single train/test split
                X_train = _RF
                y_train = RM.label.values

                X_test = _DF
                y_test = DM.label.values
                clf = linear_model.LogisticRegression(penalty='l2',C=C).fit(X_train, y_train)

                ## add prediction probabilities to metadata matrix
                cats = clf.classes_
                probs = clf.predict_proba(X_test)

                ## add prediction probabilities to metadata matrix
                ## must sort so that trained are first, and control is last
                cats = list(clf.classes_)
                _ordering = np.argsort(np.hstack((trained_objs,control_objs))) ## e.g., [chair table bench bed] ==> [3 2 0 1]
                ordering = np.argsort(_ordering) ## get indices that sort from alphabetical to (trained_objs, control_objs)
                probs = clf.predict_proba(X_test)[:,ordering] ## [table chair bed bench]
                logprobs = np.log(clf.predict_proba(X_test)[:,ordering])                               
                
                if logged==True:
                    out = logprobs
                else:
                    out = probs

                DM['t1_prob'] = out[:,0]
                DM['t2_prob'] = out[:,1]
                DM['c1_prob'] = out[:,2]
                DM['c2_prob'] = out[:,3]
                
                ## add identity of trained objects and control objects to dataframe
                DM['t1_name'] = trained_objs[0]
                DM['t2_name'] = trained_objs[1]                
                DM['c1_name'] = control_objs[0] 
                DM['c2_name'] = control_objs[1]                     

                ## also save out new columns in the same order
                if logged==True:
                    probs = np.log(clf.predict_proba(X_test))
                else:
                    probs = clf.predict_proba(X_test)
                DM['bed_prob'] = probs[:,0]
                DM['bench_prob'] = probs[:,1]
                DM['chair_prob'] = probs[:,2]
                DM['table_prob'] = probs[:,3]
                
            elif version=='4wayIndependent':

                for trained_obj in reversed(trained_objs): # reversed so that order of inclusion is t1, t2

                    inds = RM.label != trained_obj
                    _RM = RM[inds]

                    ## normalize voxels within task
                    normalize_on = 1
                    if normalize_on:
                        _RF = normalize(RF[inds,:])
                        _DF = normalize(DF)
                    else:
                        _RF = RF[inds,:]
                        _DF = DF

                    # single train/test split
                    X_train = _RF # recognition run feature set
                    y_train = _RM.label.values # list of labels for the training set

                    X_test = _DF
                    y_test = DM.label.values
                    clf = linear_model.LogisticRegression(penalty='l2',C=1).fit(X_train, y_train)

                    ## add prediction probabilities to metadata matrix
                    ## must sort so that trained are first, and control is last
                    cats = list(clf.classes_)
                    trained_index = cats.index([t for t in  trained_objs if t != trained_obj][0])
                    c1_index = cats.index(control_objs[0]) ## this is not always the target
                    c2_index = cats.index(control_objs[1]) ## this is not always the target
                    ordering = [trained_index, c1_index, c2_index]
                    probs.append(clf.predict_proba(X_test)[:,ordering])
                    logprobs.append(np.log(clf.predict_proba(X_test)[:,ordering]))

                if logged==True:
                    out = logprobs
                else:
                    out = probs
                    
                # save out new columns by object name and by t1, t2, c1, c2:
                DM['t1_prob'] = DM['{}_prob'.format(trained_objs[0])] = out[0][:,0]
                DM['t2_prob'] = DM['{}_prob'.format(trained_objs[1])] = out[1][:,0]
                DM['c1_prob'] = DM['{}_prob'.format(control_objs[0])] = (out[0][:,1] + out[1][:,1])/2.0
                DM['c2_prob'] = DM['{}_prob'.format(control_objs[0])] = (out[0][:,2] + out[1][:,2])/2.0
                
                ## add identity of trained objects and control objects to dataframe
                DM['t1_name'] = trained_objs[0]
                DM['t2_name'] = trained_objs[1]                
                DM['c1_name'] = control_objs[0] 
                DM['c2_name'] = control_objs[1]                        
                
            elif version=='3way':

                for ctrl in control_objs:

                    inds = RM.label != ctrl
                    _RM = RM[inds]

                    ## normalize voxels within task
                    normalize_on = 1
                    if normalize_on:
                        _RF = normalize(RF[inds,:])
                        _DF = normalize(DF)
                    else:
                        _RF = RF[inds,:]
                        _DF = DF

                    # single train/test split
                    X_train = _RF # recognition run feature set
                    y_train = _RM.label.values # list of labels for the training set

                    X_test = _DF
                    y_test = DM.label.values
                    clf = linear_model.LogisticRegression(penalty='l2',C=1).fit(X_train, y_train)

                    ## add prediction probabilities to metadata matrix
                    ## must sort so that trained are first, and control is last
                    cats = list(clf.classes_)
                    ctrl_index = cats.index([c for c in control_objs if c != ctrl][0])
                    t1_index = cats.index(trained_objs[0]) ## this is not always the target
                    t2_index = cats.index(trained_objs[1]) ## this is not always the target
                    ordering = [t1_index, t2_index, ctrl_index]
                    probs.append(clf.predict_proba(X_test)[:,ordering])
                    logprobs.append(np.log(clf.predict_proba(X_test)[:,ordering]))

                if logged==True:
                    out = logprobs
                else:
                    out = probs

                DM['t1_prob'] = (out[0][:,0] + out[1][:,0])/2.0
                DM['t2_prob'] = (out[0][:,1] + out[1][:,1])/2.0
                DM['c_prob'] = (out[0][:,2] + out[1][:,2])/2.0
                
                ## add identity of trained objects and control objects to dataframe
                DM['t1_name'] = trained_objs[0]
                DM['t2_name'] = trained_objs[1]                
                DM['c1_name'] = control_objs[0] 
                DM['c2_name'] = control_objs[1]                        

            elif version=='2way':

                ## subset recognition data matrices to only include the trained classes
                inds = RM.label.isin(trained_objs)
                _RM = RM[inds]

                ## normalize voxels within task
                normalize_on = 1
                if normalize_on:
                    _RF = normalize(RF[inds,:])
                    _DF = normalize(DF)
                else:
                    _RF = RF[inds,:]
                    _DF = DF

                # single train/test split
                X_train = _RF # recognition run feature set
                y_train = _RM.label.values # list of labels for the training set

                X_test = _DF
                y_test = DM.label.values
                clf = linear_model.LogisticRegression(penalty='l2',C=1).fit(X_train, y_train)

                probs = clf.predict_proba(X_test)

                ## add prediction probabilities to metadata matrix
                ## must sort so that trained are first, and control is last
                cats = list(clf.classes_)
                _ordering = np.argsort(trained_objs)
                ordering = np.argsort(_ordering)
                probs = clf.predict_proba(X_test)[:,ordering]
                logprobs = np.log(clf.predict_proba(X_test)[:,ordering])

                if logged==True:
                    out = logprobs
                else:
                    out = probs

                DM['t1_prob'] = out[:,0]
                DM['t2_prob'] = out[:,1]
                
                ## add identity of trained objects and control objects to dataframe
                DM['t1_name'] = trained_objs[0]
                DM['t2_name'] = trained_objs[1]                
                DM['c1_name'] = control_objs[0] 
                DM['c2_name'] = control_objs[1]                   

            elif version=='2wayDraw':
                INTDM = []
                __acc = []
                for i in range(1,5):
                    trainrun_inds = DM.index[DM.run_num!=i]
                    testrun_inds = DM.index[DM.run_num==i]
                    DMtrain = DM[DM.run_num!=i]
                    DMtest = DM[DM.run_num==i]
                    trainrun_feats = DF[trainrun_inds,:]
                    testrun_feats = DF[testrun_inds,:]

                    ## normalize voxels within task
                    normalize_on = 1
                    if normalize_on:
                        _DFtrain = normalize(trainrun_feats)
                        _DFtest = normalize(testrun_feats)
                    else:
                        _DFtrain = trainrun_feats
                        _DFtest = testrun_feats

                    # single train/test split
                    X_train = _DFtrain
                    y_train = DMtrain.label.values

                    X_test = _DFtest
                    y_test = DMtest.label.values

                    clf = linear_model.LogisticRegression(penalty='l2',C=1).fit(X_train, y_train)

                    probs = clf.predict_proba(X_test)

                    ## add prediction probabilities to metadata matrix
                    ## must sort so that trained are first, and control is last
                    cats = list(clf.classes_)
                    _ordering = np.argsort(trained_objs)
                    ordering = np.argsort(_ordering)
                    probs = clf.predict_proba(X_test)[:,ordering]
                    np.place(probs, probs==0, 2.22E-16)
                    #logprobs = np.log(clf.predict_proba(X_test)[:,ordering])
                    logprobs = np.log(probs)


                    if logged==True:
                        out = logprobs
                    else:
                        out = probs

                    DMtest['t1_prob'] = out[:,0]
                    DMtest['t2_prob'] = out[:,1]
                    DMtest['subj'] = np.repeat(this_sub,DMtest.shape[0])
                    DMtest['roi'] = np.repeat(this_roi,DMtest.shape[0])

                    __acc.append(clf.score(X_test, y_test))
                    if len(INTDM)==0:
                        INTDM = DMtest
                    else:
                        INTDM = pd.concat([INTDM,DMtest],ignore_index=True)
                DM = INTDM
                _acc = np.mean(np.array(__acc))

            DM['subj'] = np.repeat(this_sub,DM.shape[0])
            DM['roi'] = np.repeat(this_roi,DM.shape[0])

            if len(ALLDM)==0:
                ALLDM = DM
            else:
                ALLDM = pd.concat([ALLDM,DM],ignore_index=True)

            acc.append(_acc) if version == '2wayDraw' else acc.append(clf.score(X_test, y_test))

        Acc.append(acc)

    Acc = np.array(Acc)
    ALLDM['phase'] = 'draw'
    return ALLDM, Acc


def make_prepostrecog_predictions_withinphase(sub_list,
                                              roi_list,
                                              version='4way',
                                              test_phase='pre',
                                              logged=True,
                                              concat_initial_data=False,
                                              C=1):
    '''
    input:
        sub_list: a list containing subject IDs
        roi_list: a list containing roi names
        version: a string from options: ['4way','3way','2way']
            4way: trains to discriminate all four objects from recognition runs
        test_phase: which recognition phase to test on, "pre" or "post"
        logged: boolean. If true, return log-probabilities. If false, return raw probabilities.

    assumes: that you have directories containing recognition run and drawing run data, consisting of paired .npy
                voxel matrices and .csv metadata matrices
    '''

    ## Handle slightly different naming for same ROIs in the drawing/recog data directories

    # ROI labels in the recog data directory
    roi_list_recog = np.array(['V1Draw', 'V2Draw', 'LOCDraw', 'ParietalDraw', 
                         'supraMarginalDraw', 'postCentralDraw', 'preCentralDraw', 'FrontalDraw'])
  
    # initialize "All Data Matrix"
    ALLDM = []
    ## loop through all subjects and rois
    Acc = []
    for this_roi in roi_list:
        clear_output(wait=True)
        acc = []
        for this_sub in sub_list:
            ## load subject data in
            DM, DF = load_draw_data(this_sub,this_roi)
            RM12, RF12 = load_recog_data(this_sub,this_roi,'12')
            if test_phase=='pre':
                RM, RF = load_recog_data(this_sub,this_roi,'34')
            elif test_phase=='post':
                RM, RF = load_recog_data(this_sub,this_roi,'56')            
            else:
                print('Invalid test split, test_phase should be either "pre" or "post." ')

            ## loop through train/test split
            _acc = []
            for name, group in RM.groupby('run_num'):
                print('Now analyzing {} from {} ...'.format(this_roi, this_sub))    
                print('train run: {}, test run: {}'.format(name, np.setdiff1d([1,2],name)[0]))
                clear_output(wait=True)

                ## train/test split by run
                RMtrain = group 
                RFtrain = RF[RMtrain.index,:]
                
                if concat_initial_data:                
                    ## concat data with initial recognition run 
                    RMtrain = pd.concat([group,RM12],axis=0)
                    RFtrain = np.vstack((RFtrain,RF12))
                    ## print(RMtrain.shape, RFtrain.shape)
                
                RMtest = RM[RM['run_num']==np.setdiff1d([1,2],name)[0]] 
                RFtest = RF[RMtest.index,:]

                # identify control objects (the only use of DM in this function)
                trained_objs = np.unique(DM.label.values)
                control_objs = [i for i in ['bed','bench','chair','table'] if i not in trained_objs]

                probs = []
                logprobs = []

                ## normalize voxels within task
                normalize_on = 1
                if normalize_on:
                    _RFtrain = normalize(RFtrain)
                    _RFtest = normalize(RFtest)
                else:
                    _RFtrain = RFtrain
                    _RFtest = RFtest

                # single train/test split
                X_train = _RFtrain
                y_train = RMtrain.label.values

                X_test = _RFtest
                y_test = RMtest.label.values
                clf = linear_model.LogisticRegression(penalty='l2',C=1).fit(X_train, y_train)

                ## add prediction probabilities to metadata matrix
                cats = clf.classes_
                probs = clf.predict_proba(X_test)

                ## add prediction probabilities to metadata matrix
                ## must sort so that trained are first, and control is last
                cats = list(clf.classes_)
                _ordering = np.argsort(np.hstack((trained_objs,control_objs))) ## e.g., [chair table bench bed] ==> [3 2 0 1]
                ordering = np.argsort(_ordering) ## get indices that sort from alphabetical to (trained_objs, control_objs)
                probs = clf.predict_proba(X_test)[:,ordering] ## [table chair bed bench]
                logprobs = np.log(clf.predict_proba(X_test)[:,ordering])                               

                if logged==True:
                    out = logprobs
                else:
                    out = probs

                RM.at[RMtest.index,'t1_prob'] = out[:,0]
                RM.at[RMtest.index,'t2_prob'] = out[:,1]
                RM.at[RMtest.index,'c1_prob'] = out[:,2]   
                RM.at[RMtest.index,'c2_prob'] = out[:,3]

                ## also save out new columns in the same order
                if logged==True:
                    probs = np.log(clf.predict_proba(X_test))
                else:
                    probs = clf.predict_proba(X_test)
                RM.at[RMtest.index,'bed_prob'] = probs[:,0]
                RM.at[RMtest.index,'bench_prob'] = probs[:,1]
                RM.at[RMtest.index,'chair_prob'] = probs[:,2]
                RM.at[RMtest.index,'table_prob'] = probs[:,3]

                ## add identity of trained objects and control objects to dataframe
                RM.at[RMtest.index,'t1_name'] = trained_objs[0]
                RM.at[RMtest.index,'t2_name'] = trained_objs[1]
                RM.at[RMtest.index,'c1_name'] = control_objs[0]
                RM.at[RMtest.index,'c2_name'] = control_objs[1]            

                RM.at[RMtest.index,'subj'] = np.repeat(this_sub,len(RMtest.index))
                RM.at[RMtest.index,'roi'] = np.repeat(this_roi,len(RMtest.index))   
                
                _acc.append(clf.score(X_test, y_test))

            ## this is part of the subject loop, do this once per subject
            if len(ALLDM)==0:
                ALLDM = RM
            else:
                ALLDM = pd.concat([ALLDM,RM],ignore_index=True)

                            
            acc.append(np.mean(_acc))
        Acc.append(acc)
    Acc = np.array(Acc)
    ALLDM['phase'] = test_phase
    return ALLDM, Acc


def make_prepostrecog_predictions(sub_list,
                                  roi_list,
                                  version='4way',
                                  test_phase='pre',
                                  logged=True,
                                  C=1):
    '''
    input:
        sub_list: a list containing subject IDs
        roi_list: a list containing roi names
        version: a string from options: ['4way','3way','2way']
            4way: trains to discriminate all four objects from recognition runs
        test_phase: which recognition phase to test on, "pre" or "post"
        logged: boolean. If true, return log-probabilities. If false, return raw probabilities.

    assumes: that you have directories containing recognition run and drawing run data, consisting of paired .npy
                voxel matrices and .csv metadata matrices
    '''

    ## Handle slightly different naming for same ROIs in the drawing/recog data directories

    # ROI labels in the recog data directory
    roi_list_recog = np.array(['V1Draw', 'V2Draw', 'LOCDraw', 'ParietalDraw', 
                         'supraMarginalDraw', 'postCentralDraw', 'preCentralDraw', 'FrontalDraw'])
  
    # initialize "All Data Matrix"
    ALLDM = []
    ## loop through all subjects and rois
    Acc = []
    for this_roi in roi_list:
        print('Now analyzing {} ...'.format(this_roi))
        clear_output(wait=True)
        acc = []
        for this_sub in sub_list:
            ## load subject data in
            ## "localizer"
            RM, RF = load_recog_data(this_sub,this_roi,'12')
            DM, DF = load_draw_data(this_sub,this_roi)
            if test_phase=='pre':
                RMtest, RFtest = load_recog_data(this_sub,this_roi,'34')
            elif test_phase=='post':
                RMtest, RFtest = load_recog_data(this_sub,this_roi,'56')            
            else:
                print('Invalid test split, test_phase should be either "pre" or "post." ')
            # identify control objects;
            # we wil train one classifier with
            
            trained_objs = np.unique(DM.label.values)
            control_objs = [i for i in ['bed','bench','chair','table'] if i not in trained_objs]
            
            probs = []
            logprobs = []

            if version=='4way':
                ## normalize voxels within task
                normalize_on = 1
                if normalize_on:
                    _RF = normalize(RF)
                    _RFtest = normalize(RFtest)
                else:
                    _RF = RF
                    _RFtest = RFtest

                # single train/test split
                X_train = _RF
                y_train = RM.label.values

                X_test = _RFtest
                y_test = RMtest.label.values
                clf = linear_model.LogisticRegression(penalty='l2',C=1).fit(X_train, y_train)

                ## add prediction probabilities to metadata matrix
                cats = clf.classes_
                probs = clf.predict_proba(X_test)

                ## add prediction probabilities to metadata matrix
                ## must sort so that trained are first, and control is last
                cats = list(clf.classes_)
                _ordering = np.argsort(np.hstack((trained_objs,control_objs))) ## e.g., [chair table bench bed] ==> [3 2 0 1]
                ordering = np.argsort(_ordering) ## get indices that sort from alphabetical to (trained_objs, control_objs)
                probs = clf.predict_proba(X_test)[:,ordering] ## [table chair bed bench]
                logprobs = np.log(clf.predict_proba(X_test)[:,ordering])                               
                
                if logged==True:
                    out = logprobs
                else:
                    out = probs

                RMtest['t1_prob'] = out[:,0]
                RMtest['t2_prob'] = out[:,1]
                RMtest['c1_prob'] = out[:,2]
                RMtest['c2_prob'] = out[:,3]

                ## also save out new columns in the same order
                if logged==True:
                    probs = np.log(clf.predict_proba(X_test))
                else:
                    probs = clf.predict_proba(X_test)
                RMtest['bed_prob'] = probs[:,0]
                RMtest['bench_prob'] = probs[:,1]
                RMtest['chair_prob'] = probs[:,2]
                RMtest['table_prob'] = probs[:,3]
                
                ## add identity of trained objects and control objects to dataframe
                RMtest['t1_name'] = trained_objs[0]
                RMtest['t2_name'] = trained_objs[1]                
                RMtest['c1_name'] = control_objs[0] 
                RMtest['c2_name'] = control_objs[1]             

            RMtest['subj'] = np.repeat(this_sub,RMtest.shape[0])
            RMtest['roi'] = np.repeat(this_roi,RMtest.shape[0])

            if len(ALLDM)==0:
                ALLDM = RMtest
            else:
                ALLDM = pd.concat([ALLDM,RMtest],ignore_index=True)

            acc.append(_acc) if version == '2wayDraw' else acc.append(clf.score(X_test, y_test))

        Acc.append(acc)
        
    Acc = np.array(Acc)
    return ALLDM, Acc


def plot_summary_timecourse(ALLDM, 
                            this_iv='trial_num',
                            roi_list=['V1','V2','LOC'],
                            render_cond=1,
                            version='4way',
                            proj_dir='../',
                            baseline_correct=False,
                            nb_name='2_object_evidence_during_drawing'):
    '''
    input: 
        ALLDM matrix: supersubject matrix generated by fn make_drawing_predictions
        this_iv: choice of time binning options. options are ['time_point','trial_num','run_num']
        roi_list: list of ROIs to make plots for
        render_cond: Is 1 if you want to the CONDITION-wise plots -- trained vs. foil vs control
                     Is 0 if if you want the DIFFERENCE plots -- trained - foil vs foil - control
        version: Using 4-way, 3-way, or 2-way classifier results? options are ['2way','3way','4way']
        baseline_correct: If you want to subtract the first observation from the time course
        nb_name: which notebook is this from
        proj_dir: root directory of project.
    
    output: 
        saves PNG versions of plots in plots dir, which is located within results dir, which is
        itself at top of project hierarchy
    '''    
    
    subs = np.unique(ALLDM.subj.values)
    lookup = dict(zip(['trial_num','run_num','time_point'],['repetition','run','TR']))

    ivs=['run_num','trial_num','time_point']
    assert this_iv in ivs    
    

    for this_roi in roi_list:
        
        print('Now plotting results for {} ...'.format(this_roi))

        T = []
        F = []
        C = []
        Sub = []
        for sub in subs:
            inds = (ALLDM['roi']==this_roi) & (ALLDM['subj']==sub) if this_roi != 'VGG' else (ALLDM['roi']==this_roi) & (ALLDM['subj']==sub) & (ALLDM['time_point'] == 23)
            t,f,c = get_prob_timecourse(this_iv,ALLDM[inds],version=version)
            if baseline_correct:
                t = t - t[0]
                f = f - f[0]
                c = c - c[0]
            if len(T)==0:
                T = t
                F = f
                C = c
                DTF = t-f  ## these differences already in log space       
                DTC = t-c
                DFC = f-c
            else:
                T = np.hstack((T,t))
                F = np.hstack((F,f))        
                C = np.hstack((C,c)) 
                DTF = np.hstack((DTF,t-f))                
                DTC = np.hstack((DTC,t-c))
                DFC = np.hstack((DFC,f-c))
            Sub.append([sub]*len(t))   

        if render_cond==1:
            ## make longform version of dataframe to use in tsplot (by condition)            
            Trial = np.tile(np.arange(len(t)),len(subs)*3)
            Condition = np.repeat(['target','foil','control'],len(T))
            Sub = np.tile(np.array(flatten(Sub)),3)
            Prob = np.hstack((T,F,C))
            assert len(Trial)==len(Condition)
            assert len(Sub)==len(Prob)
            assert len(Condition)==len(Sub)
            x = pd.DataFrame([Prob,Trial,Condition,Sub])
            x = x.transpose()
            x.columns = ['probability',lookup[this_iv],'condition','sub']
            toop = 'condition'            
        else:
            ## make longform version of dataframe to use in tsplot (difference btw conditions)                    
            Trial = np.tile(np.arange(len(t)),len(subs)*3)
            Condition = np.repeat(['target-foil','target-control','foil-control'],len(T))
            Sub = np.tile(np.array(flatten(Sub)),3)
            Prob = np.hstack((DTF,DTC,DFC))        
            assert len(Trial)==len(Condition)
            assert len(Sub)==len(Prob)
            assert len(Condition)==len(Sub)
            x = pd.DataFrame([Prob,Trial,Condition,Sub])
            x = x.transpose()
            x.columns = ['probability',lookup[this_iv],'condition','sub']        
            toop = 'difference'
        #print(x)   
        fig = plt.figure(figsize=(8,4)) 
        plt.subplot(111)
        ## plot it
        color_picker = ['#dd4318','#0d61c6','#4a4b4c']
        sns.set_palette(color_picker)
        x['timePlusOne'] = x[lookup[this_iv]].apply(lambda x: x+1)
        sns.tsplot(data=x,
                  time='timePlusOne',
                  unit='sub',
                  condition='condition',
                  value='probability',
                  ci=95)
        if render_cond==1:
            plt.ylim(0,0.5)
            plt.axhline(0.25,linestyle=':',color='k')  
            plt.legend(bbox_to_anchor=(1.01, 0.8))  
            plt.gca().get_legend().remove()
            plt.title('object evidence in {}'.format(this_roi))

        else:
            plt.ylim(-0.3,0.3)
            plt.axhline(0,linestyle=':',color='k')  
            plt.legend(bbox_to_anchor=(0.7, 1.01))                        
            plt.title('difference in classifier evidence by condition in {}'.format(this_roi))             
        plt.xlabel(lookup[this_iv])
        plt.xticks(np.arange(1,np.max(x['timePlusOne'].values)+1))
        plt.tick_params(axis='both', which='major', labelsize=14)
        if not os.path.exists(os.path.join(plot_dir,'{}/{}/{}'.format(nb_name,lookup[this_iv],toop))):
            os.makedirs(os.path.join(plot_dir,'{}/{}/{}'.format(nb_name,lookup[this_iv],toop)))
        plt.tight_layout(rect=[0,0,1,0.7])
        plt.savefig(os.path.join(plot_dir,'{}/{}/{}/prob_timecourse_{}_by_{}_{}.pdf'.\
                    format(nb_name,lookup[this_iv],toop,this_roi,lookup[this_iv],version)))
        plt.close(fig)    

        
def get_log_odds(ALLDM,
                 this_iv = 'trial_num',
                 roi_list = ['V1','V2'],
                 phase = 'NOPHASE',
                 version='4way',
                 logged=True,
                 proj_dir='../'):
    '''
    input: 
        ALLDM
        this_iv: options are ['run_num','trial_num','time_point']
        roi_list: list of ROIs
        version: which N-way classifier ['2way','3way','4way']
        logged: True if using log probabilities to compute odds, False if not
        proj_dir: path to root of project directory
    output: 
        d: pandas dataframe containing difference in log probabilities (log odds)
        CSV files with prefix "difference_logprobs"
        prints log odds to console
    '''

    sub_tf = []
    sub_tc = []
    sub_fc = []
    roi = []
    
    subs = np.unique(ALLDM['subj'].values)
    lookup = dict(zip(['trial_num','run_num','time_point'],['repetition','run','TR']))
    
    for this_roi in roi_list:
        T = []
        F = []
        C = []
        Sub = []
        for sub in subs:
            print('Now analyzing {} from {}...'.format(this_roi,sub))
            clear_output(wait=True)
            inds = (ALLDM['roi']==this_roi) & (ALLDM['subj']==sub) 
            t,f,c = get_log_prob_timecourse(this_iv,ALLDM[inds],version=version)
            if len(T)==0:
                T = t
                F = f
                C = c
                DTF = t-f               
                DTC = t-c
                DFC = f-c
            else:
                T = np.hstack((T,t))
                F = np.hstack((F,f))        
                C = np.hstack((C,c)) 
                DTF = np.hstack((DTF,t-f))                
                DTC = np.hstack((DTC,t-c))
                DFC = np.hstack((DFC,f-c))
            Sub.append([sub]*len(t))   

        ## make longform version of dataframe to use in tsplot (difference btw conditions)                    
        Trial = np.tile(np.arange(len(t)),len(subs)*3)
        Condition = np.repeat(['target-foil','target-control','foil-control'],len(T))
        Sub = np.tile(np.array(flatten(Sub)),3)
        Prob = np.hstack((DTF,DTC,DFC))        
        assert len(Trial)==len(Condition)
        assert len(Sub)==len(Prob)
        assert len(Condition)==len(Sub)
        x = pd.DataFrame([Prob,Trial,Condition,Sub])
        x = x.transpose()
        x.columns = ['probability',lookup[this_iv],'condition','sub']

        for this_sub in subs:
            sub_tf.append(x[(x['condition']=='target-foil') & (x['sub']==this_sub)]['probability'].mean())
            sub_tc.append(x[(x['condition']=='target-control') & (x['sub']==this_sub)]['probability'].mean())  
            sub_fc.append(x[(x['condition']=='foil-control') & (x['sub']==this_sub)]['probability'].mean()) 
            roi.append(this_roi)

        ## save out big dataframe with all subjects and timepoints
        x.to_csv(os.path.join(proj_dir, 'csv/roi/object_classifier_logprobs_{}_{}_{}.csv'.format(phase,this_roi,this_iv)),index=False)

    ## make dataframe with subject-level difference scores
    substr = [str(i).zfill(7) for i in subs]
    flat_sub_list = flatten([substr]*len(roi_list_recog))
    assert len(flat_sub_list)==len(roi)
    d = pd.DataFrame([sub_tf,sub_tc,sub_fc,roi,flat_sub_list])
    d = d.transpose()
    d.columns = ['target-foil','target-control','foil-control','roi','sub']
    d = d.astype({'target-foil':'float64','target-control':'float64',
                  'foil-control':'float64','sub':'str'})

    ## output target-foil ratios
    if logged==True:
        print(d.groupby('roi')['target-foil'].apply(lambda x: np.mean(np.exp(x))))
        d.to_csv(os.path.join(proj_dir, 'csv/object_classifier_logodds_{}.csv'.format(phase)),index=False)
    else:
        print(d.groupby('roi')['target-foil'].mean())
        d.to_csv(os.path.join(proj_dir, 'csv/object_classifier_rawprobs_{}.csv'.format(phase)),index=False)
        
    return d        
        
def get_log_prob_timecourse(iv,DM,version='4way'):

    t1 = DM['t1_name'].unique()[0]
    t2 = DM['t2_name'].unique()[0]
    c1 = DM['c1_name'].unique()[0]
    c2 = DM['c2_name'].unique()[0]   

    
    if DM['phase'].unique()[0] in ['pre','post']: ## then this is a recog run, so log prob timecourse is computed differently
        target = np.hstack((DM[DM.label==t1]['t1_prob'].values,DM[DM.label==t2]['t2_prob'].values))
        foil = np.hstack((DM[DM.label==t1]['t2_prob'].values,DM[DM.label==t2]['t1_prob'].values))
        c1 = np.hstack((DM[DM.label==t1]['c1_prob'].values,DM[DM.label==t2]['c1_prob'].values))
        c2 = np.hstack((DM[DM.label==t1]['c2_prob'].values,DM[DM.label==t2]['c2_prob'].values))
        control = np.vstack((c1,c2)).mean(0)    
    
    elif version[:4]=='4way': ## assuming that this is a drawing run             
        
        target = np.vstack((DM[DM.label==t1].groupby(iv)['t1_prob'].mean().values,
                       DM[DM.label==t2].groupby(iv)['t2_prob'].mean().values)).mean(0) ## target timecourse
        foil = np.vstack((DM[DM.label==t1].groupby(iv)['t2_prob'].mean().values,
                       DM[DM.label==t2].groupby(iv)['t1_prob'].mean().values)).mean(0) ## foil timecourse
        control = np.vstack((DM[DM.label==t1].groupby(iv)['c1_prob'].mean().values,
                            DM[DM.label==t1].groupby(iv)['c2_prob'].mean().values,
                            DM[DM.label==t2].groupby(iv)['c1_prob'].mean().values,
                            DM[DM.label==t2].groupby(iv)['c2_prob'].mean().values)).mean(0) ## control timecourse
    elif version[:4]=='3way':
        target = np.vstack((DM[DM.label==t1].groupby(iv)['t1_prob'].mean().values,
                       DM[DM.label==t2].groupby(iv)['t2_prob'].mean().values)).mean(0) ## target timecourse; mean is taken over what?
        foil = np.vstack((DM[DM.label==t1].groupby(iv)['t2_prob'].mean().values,
                       DM[DM.label==t2].groupby(iv)['t1_prob'].mean().values)).mean(0) ## foil timecourse
        control = np.vstack((DM[DM.label==t1].groupby(iv)['c_prob'].mean().values,
                            DM[DM.label==t2].groupby(iv)['c_prob'].mean().values)).mean(0) ## control timecourse

    elif version[:4]=='2way':
        target = np.vstack((DM[DM.label==t1].groupby(iv)['t1_prob'].mean().values,
                       DM[DM.label==t2].groupby(iv)['t2_prob'].mean().values)).mean(0) ## target timecourse; mean is taken over what?
        foil = np.vstack((DM[DM.label==t1].groupby(iv)['t2_prob'].mean().values,
                       DM[DM.label==t2].groupby(iv)['t1_prob'].mean().values)).mean(0) ## foil timecourse

        control = np.zeros(len(foil))

    return target, foil, control 


## plotting helper
def get_prob_timecourse(iv,DM,version='4way'):
    trained_objs = np.unique(DM.label.values)
    control_objs = [i for i in ['bed','bench','chair','table'] if i not in trained_objs]
    
    DM.rename(columns={'t1_prob':'t1_logprob','t2_prob':'t2_logprob',
                      'c1_prob':'c1_logprob','c2_prob':'c2_logprob'},inplace=True)
    t1_prob = np.exp(DM['t1_logprob']).values
    DM = DM.assign(t1_prob=pd.Series(t1_prob).values)
    t2_prob = np.exp(DM['t2_logprob']).values
    DM = DM.assign(t2_prob=pd.Series(t2_prob).values)
    c1_prob = np.exp(DM['c1_logprob']).values
    DM = DM.assign(c1_prob=pd.Series(c1_prob).values)
    c2_prob = np.exp(DM['c2_logprob']).values
    DM = DM.assign(c2_prob=pd.Series(c2_prob).values)    

    if version[:4]=='4way':
        t1 = trained_objs[0]
        t2 = trained_objs[1]
        c1 = control_objs[0]
        c2 = control_objs[1]
        target = np.vstack((DM[DM.label==t1].groupby(iv)['t1_prob'].mean().values,
                       DM[DM.label==t2].groupby(iv)['t2_prob'].mean().values)).mean(0) ## target timecourse
        foil = np.vstack((DM[DM.label==t1].groupby(iv)['t2_prob'].mean().values,
                       DM[DM.label==t2].groupby(iv)['t1_prob'].mean().values)).mean(0) ## foil timecourse
        control = np.vstack((DM[DM.label==t1].groupby(iv)['c1_prob'].mean().values,
                            DM[DM.label==t1].groupby(iv)['c2_prob'].mean().values,
                            DM[DM.label==t2].groupby(iv)['c1_prob'].mean().values,
                            DM[DM.label==t2].groupby(iv)['c2_prob'].mean().values)).mean(0) ## control timecourse
    elif version[:4]=='3way':
        t1 = trained_objs[0]
        t2 = trained_objs[1]
        target = np.vstack((DM[DM.label==t1].groupby(iv)['t1_prob'].mean().values,
                       DM[DM.label==t2].groupby(iv)['t2_prob'].mean().values)).mean(0) ## target timecourse; mean is taken over what?
        foil = np.vstack((DM[DM.label==t1].groupby(iv)['t2_prob'].mean().values,
                       DM[DM.label==t2].groupby(iv)['t1_prob'].mean().values)).mean(0) ## foil timecourse
        control = np.vstack((DM[DM.label==t1].groupby(iv)['c_prob'].mean().values,
                            DM[DM.label==t2].groupby(iv)['c_prob'].mean().values)).mean(0) ## control timecourse

    elif version[:4]=='2way':
        t1 = trained_objs[0]
        t2 = trained_objs[1]
        target = np.vstack((DM[DM.label==t1].groupby(iv)['t1_prob'].mean().values,
                       DM[DM.label==t2].groupby(iv)['t2_prob'].mean().values)).mean(0) ## target timecourse; mean is taken over what?
        foil = np.vstack((DM[DM.label==t1].groupby(iv)['t2_prob'].mean().values,
                       DM[DM.label==t2].groupby(iv)['t1_prob'].mean().values)).mean(0) ## foil timecourse

        control = np.zeros(len(foil))

    return target, foil, control
        
def preprocess_acc_array(Acc,phase='draw', 
                         roi_list = ['V1', 'V2', 'LOC', 'FUS', 'PHC', 'IT', 'ENT', 'PRC', 'HC'],
                         sub_list = ['0110171', '0110172', '0111171', '0112171', '0112172', '0112173',
                                   '0113171', '0115174', '0117171', '0118171', '0118172', '0119171',
                                   '0119172', '0119173', '0119174', '0120171', '0120172', '0120173',
                                   '0123171', '0123173', '0124171', '0125171', '0125172', '1121161',
                                   '1130161', '1202161', '1203161', '1206161', '1206162', '1206163',
                                   '1207162']):
    '''
    called in notebook 1_object_evidence_during_recognition
    '''
    
    A = pd.DataFrame(Acc.T)
    A.columns = roi_list
    A['sub'] = sub_list
    A['phase'] = phase
    A1 = pd.melt(A, 
                id_vars=['sub','phase'], 
                var_name='roi',
                value_vars=roi_list, 
                value_name='acc') 
    return A1        
        
def add_target_prob_column(df):
    '''
    df is dataframe, e.g., ALLPRE or ALLPOST that contains classifier probabilities for recognition runs
    in either the pre or post phases, respectively.

    '''
    
    df['target_prob_raw'] = np.nan
    df['trained'] = np.bool

    for ind,d in df.iterrows():
        print('Analyzing {} of {}'.format(ind,df.shape[0]))
        clear_output(wait=True)
        ordered_entries = ['t1_name','t2_name','c1_name','c2_name']
        ordered_labels = d[['t1_name','t2_name','c1_name','c2_name']].values.tolist()
        obj2cond = dict(zip(ordered_labels,ordered_entries))
        this_obj = d['label']
        this_raw_column = '{}_prob_raw'.format(obj2cond[this_obj].split('_')[0])

        ## assign target probability (raw space) to dedicated column
        df.at[ind,'target_prob_raw'] = d[this_raw_column]

        ## assign condition of stimulus
        cond = True if obj2cond[this_obj][0]=='t' else False
        df.at[ind,'trained'] = cond  
    
        ## if trained object, also extract the foil probability
        foil_mapper = dict(zip(['t1','t2','c1','c2'],['t2','t1','c2','c1']))        
        foil_id = foil_mapper[obj2cond[this_obj].split('_')[0]]
        df.at[ind,'foil_prob_raw'] = d['{}_prob_raw'.format(foil_id)]       
    
    return df

def postprocess_prepost_rawprobs(prepost):
    
    '''
    computes variuos pre-post learning-related metrics for trained objects 
    only applied to recognition trials for trained objects    
    '''
    
    x = prepost[prepost['trained']==True].groupby(['roi_formatted','subj','phase'])['target_prob_raw'].mean().reset_index()
    x['foil_prob_raw'] = prepost[prepost['trained']==True].groupby(['roi_formatted','subj','phase'])['foil_prob_raw'].mean().reset_index()['foil_prob_raw']
    ## split then splice together so that each row contains subject's own pre and post probs, then add raw prob difference column
    x2 = x[x['phase']=='pre'].reset_index().join(x[x['phase']=='post'].reset_index(),rsuffix='_post', lsuffix='_pre')

    ######### raw probabilities ########
    ## raw prob prepost changes within target, and within foil
    x2['target_rawprob_postpre'] = x2['target_prob_raw_post'] - x2['target_prob_raw_pre']
    x2['foil_rawprob_postpre'] = x2['foil_prob_raw_post'] - x2['foil_prob_raw_pre']

    ## raw probs target vs. foil in each phase
    x2['target_foil_rawprob_pre'] = x2['target_prob_raw_pre'] - x2['foil_prob_raw_pre']
    x2['target_foil_rawprob_post'] = x2['target_prob_raw_post'] - x2['foil_prob_raw_post']

    ## raw probs prepost change in target vs. foil
    x2['target_foil_rawprob_postpre'] =  x2['target_foil_rawprob_post'] - x2['target_foil_rawprob_pre']

    ######### log odds ########
    ## get log probs and get log odds column
    x2['target_logprob_pre'] = np.log(x2['target_prob_raw_pre'])
    x2['target_logprob_post'] = np.log(x2['target_prob_raw_post'])
    x2['foil_logprob_pre'] = np.log(x2['foil_prob_raw_pre'])
    x2['foil_logprob_post'] = np.log(x2['foil_prob_raw_post'])

    ## log odds target vs. foil in each phase
    x2['target_foil_logodds_pre'] = x2['target_logprob_pre'] - x2['foil_logprob_pre']
    x2['target_foil_logodds_post'] = x2['target_logprob_post'] - x2['foil_logprob_post']

    ## log odds for change within target, and within foil
    x2['target_logodds_postpre'] = x2['target_logprob_post'] - x2['target_logprob_pre']
    x2['foil_logodds_postpre'] = x2['foil_logprob_post'] - x2['foil_logprob_pre']

    ## log odds prepost change in target vs. foil
    x2['target_foil_logodds_postpre'] = x2['target_logodds_postpre'] - x2['foil_logodds_postpre']
    
    return x2

def resample_subs(D,
                  groupby=['roi'],
                  random_state=0):
    
    Dboot = D.groupby(['roi']).apply(lambda x: x.sample(n=len(x), replace=True, random_state=random_state))
    cols = Dboot.columns
    Dboot = Dboot.xs(cols,axis=1,drop_level=True).reset_index(drop=True)
    return Dboot

def get_corr(x,y,rounding=5):
    return np.round(stats.pearsonr(x,y)[0],rounding)

def get_ci_bounds(x):
    lb = np.round(np.percentile(x,2.5),5)
    ub = np.round(np.percentile(x,97.5),5)
    return (lb,ub)

def make_drawing_connectivity_predictions(sub_list, roi_list, version='phase', feature_type='connect', logged=True):
    
    '''
    input:
        sub_list: a list containing subject IDs
        roi_list: a list containing roi names
        version: a string from options: ['phase','allruns']
            phase: trains and tests within early production runs, or late production runs 
                (i.e. train on run 1, test on run 2 and vice versa)
            allruns: trains on 3 runs, tests on a held out fourth (i.e. train on runs 1, 3 and 4, test on run 2).
        feature_type: a string from options: ['connect', 'stacked']
        logged: boolean. If true, return log-probabilities. If false, return raw probabilities.
    assumes: that you have a directory containing trial-wise drawing run connectivity data, consisting 
                    of paired .npy voxel matrices and .csv metadata matrices
    '''
    ALLDM = []
    # loop through all subjects and roi pairs
    Acc = []
    all_pairs = list(itertools.combinations(roi_list, 2))
    for (this_roi, that_roi) in all_pairs:
        print('Now analyzing {}, {} ...'.format(this_roi, that_roi))
        acc = []
        for this_sub in sub_list:
            print(this_sub)
            ## load subject data in
            DM, DF = load_connect_data(this_sub, str(this_roi)+'_'+str(that_roi), feature_type)

            trained_objs = np.unique(DM.label.values)

            INTDM = []
            __acc = []

            compare = [2,1,4,3]
            for i in range(1,5):
                if version == 'phase':
                    trainrun_inds = DM.index[DM.run_num == compare[i-1]]
                    DMtrain = DM[DM.run_num == compare[i-1]]
                elif version == 'allruns':
                    trainrun_inds = DM.index[DM.run_num != i]
                    DMtrain = DM[DM.run_num != i]
                testrun_inds = DM.index[DM.run_num == i]
                DMtest = DM[DM.run_num == i]
                trainrun_feats = DF[trainrun_inds, :]
                testrun_feats = DF[testrun_inds, :]

                # normalize voxels within task
                normalize_on = 1
                if normalize_on:
                    _DFtrain = normalize(trainrun_feats)
                    _DFtest = normalize(testrun_feats)
                else:
                    _DFtrain = trainrun_feats
                    _DFtest = testrun_feats

                X_train = _DFtrain
                y_train = DMtrain.label.values

                X_test = _DFtest
                y_test = DMtest.label.values

                clf = linear_model.LogisticRegression(penalty='l2', C=1).fit(X_train, y_train)

                ## add prediction probabilities to metadata matrix
                _ordering = np.argsort(trained_objs)
                ordering = np.argsort(_ordering)
                probs = clf.predict_proba(X_test)[:, ordering]
                # replace 0 values
                np.place(probs, probs == 0, 2.22E-16)
                logprobs = np.log(probs)

                if logged == True:
                    out = logprobs
                else:
                    out = probs

                DMtest['t1_prob'] = out[:, 0]
                DMtest['t2_prob'] = out[:, 1]
                DMtest['subj'] = np.repeat(this_sub, DMtest.shape[0])
                DMtest['roi1'] = np.repeat(this_roi, DMtest.shape[0])
                DMtest['roi2'] = np.repeat(that_roi, DMtest.shape[0])

                __acc.append(clf.score(X_test, y_test))
                if len(INTDM) == 0:
                    INTDM = DMtest
                else:
                    INTDM = pd.concat([INTDM, DMtest], ignore_index=True)

            DM = INTDM
            _acc = np.mean(np.array(__acc))

            DM['subj'] = np.repeat(this_sub, DM.shape[0])

            if len(ALLDM) == 0:
                ALLDM = DM
            else:
                ALLDM = pd.concat([ALLDM, DM], ignore_index=True)

            acc.append(_acc)
        Acc.append(acc)
    return ALLDM, Acc

def get_connect_timecourse(iv,DM):
    trained_objs = np.unique(DM.label.values)
    
    t1 = trained_objs[0]
    t2 = trained_objs[1]
    target = np.vstack((DM[DM.label==t1].groupby(iv)['t1_prob'].mean().values,
                        DM[DM.label==t2].groupby(iv)['t2_prob'].mean().values)).mean(0) ## target timecourse
    foil = np.vstack((DM[DM.label==t1].groupby(iv)['t2_prob'].mean().values,
                      DM[DM.label==t2].groupby(iv)['t1_prob'].mean().values)).mean(0) ## foil timecourse
    return target, foil

def plot_connect_timecourse(ALLDM, 
                            this_iv='trial_num',
                            roi_list=['V1Draw','V2Draw','LOCDraw'],
                            render_cond=1,
                            version='phase',
                            feature_type='connect',
                            logged=True,
                            proj_dir='../',
                            baseline_correct=False,
                            nb_name='4_connectivity_pattern_during_drawing',
                            plotType='bar'):
    '''
    input: 
        ALLDM matrix: supersubject matrix generated by fn make_drawing_connectivity_predictions
        this_iv: choice of time binning options. options are ['trial_num','run_num', 'phase_num']
        roi_list: list of ROIs to make plots for
        render_cond: Is 1 if you want to the CONDITION-wise plots -- trained vs. foil
                     Is 0 if if you want the DIFFERENCE plots -- trained - foil
        version: Using results from phasewise classification, or allruns? options are ['phase','allruns']
        baseline_correct: If you want to subtract the first observation from the time course
        nb_name: which notebook is this from
        proj_dir: root directory of project.
    
    output: 
        saves PDF versions of plots in plots dir, which is located at top level of project directory
    '''    
    
    subs = np.unique(ALLDM.subj.values)

    lookup = dict(zip(['trial_num','run_num','phase_num'],['repetition','run','phase']))

    ivs=['run_num','trial_num', 'phase_num']
    assert this_iv in ivs    
    all_pairs = list(itertools.combinations(roi_list, 2))

    for (this_roi, that_roi) in all_pairs:
       
        print('Now plotting results for {} and {} ...'.format(this_roi, that_roi))

        T = []
        F = []
        C = []
        Sub = []
        conds = ['Target - Foil','Target','Foil']
        colors = [(199/255,139/255,234/255),(241/255,112/255,13/255),(125/255,162/255,197/255)]
        for sub in subs:
            inds = (ALLDM['roi1']==this_roi) & (ALLDM['roi2']==that_roi) & (ALLDM['subj']==sub)
            t,f = get_connect_timecourse(this_iv,ALLDM[inds])
            if baseline_correct:
                t = t - t[0]
                f = f - f[0]
            if len(T)==0:
                T = t
                F = f
                DTF = t-f  ## these differences already in log space       
            else:
                T = np.hstack((T,t))
                F = np.hstack((F,f))        
                DTF = np.hstack((DTF,t-f))                
            Sub.append([sub]*len(t))   

        if render_cond==1:
            ## make longform version of dataframe to use in tsplot (by condition)            
            Trial = np.tile(np.arange(len(t)),len(subs)*2)
            Condition = np.repeat(['Target','Foil'],len(T))
            Sub = np.tile(np.array(flatten(Sub)),2)
            Prob = np.hstack((T,F))
            assert len(Trial)==len(Condition)
            assert len(Sub)==len(Prob)
            assert len(Condition)==len(Sub)
            x = pd.DataFrame([Prob,Trial,Condition,Sub])
            x = x.transpose()
            x.columns = ['probability',lookup[this_iv],'condition','sub']
            toop = 'condition'            
        else:
            ## make longform version of dataframe to use in tsplot (difference btw conditions)                    
            Trial = np.tile(np.arange(len(t)),len(subs))
            Condition = np.repeat(['Target - Foil'],len(T))
            Sub = np.tile(np.array(flatten(Sub)),1)
            Prob = DTF
            assert len(Trial)==len(Condition)
            assert len(Sub)==len(Prob)
            assert len(Condition)==len(Sub)
            x = pd.DataFrame([Prob,Trial,Condition,Sub])
            x = x.transpose()
            x.columns = ['probability',lookup[this_iv],'condition','sub']        
            toop = 'difference'
            x.to_csv('{}/{}_{}_{}_{}.csv'.format(results_dir, feature_type, this_roi, that_roi, version))
        fig, ax = plt.subplots(figsize=(5, 5))
        ## plot it
        if plotType == 'line':
            sns.tsplot(data=x,
                      time=lookup[this_iv],
                      unit='sub',
                      condition='condition',
                      color=dict(zip(conds, colors)),
                      value='probability',
                      ci=95,
                      lw=5,
                      marker='o', 
                      err_style=['ci_bars'])
            plt.xticks([-0.25,0,1,1.25], ['',1,2,''], fontsize=20, **{'fontname':'Arial Narrow'})
        if plotType == 'bar':
            if render_cond == 1:
                print('ERROR: cannot create barplot with this render cond')
                return False
            data=x[x['condition'] == 'Target - Foil']
            sns.barplot(x=lookup[this_iv],
                        y='probability',
                        data=x[x['condition'] == 'Target - Foil'],
                        color=colors[0],
                        ci=95)
            for patch in ax.patches:
                currwidth = patch.get_width()
                diff = currwidth - 0.5
                patch.set_width(0.5)
                patch.set_x(patch.get_x() + diff * .5)
            plt.xticks([-0.5,0,1,1.5], ['',1,2,''], fontsize=20, **{'fontname':'Arial Narrow'})
        if render_cond==1:
            plt.ylim(-3,-0.5)
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0.)  
        else:
            plt.ylim(0,2)
            plt.yticks(np.arange(0, 1.6, 0.5), np.arange(0, 1.6, 0.5), fontsize=20, **{'fontname':'Arial Narrow'})
            plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3, ncol=3, mode="expand", borderaxespad=0., fontsize=20)                        
        plt.xlabel('Phase Number', fontsize=20, **{'fontname':'Arial Narrow'})
        plt.ylabel('Log Probability', fontsize=20, **{'fontname':'Arial Narrow'})
        if not os.path.exists(os.path.join(plot_dir,'{}/{}/{}'.format(nb_name,lookup[this_iv],toop))):
            os.makedirs(os.path.join(plot_dir,'{}/{}/{}'.format(nb_name,lookup[this_iv],toop)))
        plt.tight_layout()        
        plt.savefig(os.path.join(plot_dir,'{}/{}/{}/{}_timecourse_{}_{}_by_{}_{}.pdf'.\
                    format(nb_name,lookup[this_iv],toop,feature_type,this_roi,that_roi,lookup[this_iv],version)))
        
