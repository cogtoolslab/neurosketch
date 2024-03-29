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

from scipy.misc import imread, imresize
from scipy.stats import norm, linregress
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib import rcParams
rcParams['pdf.fonttype'] = 42
rcParams.update({'figure.autolayout': True})   
import seaborn as sns
from IPython.display import clear_output
sns.set_context('talk')
colors = sns.color_palette("cubehelix", 5)


###############################################################################################
################### GLOBALS ###################################################################
###############################################################################################
curr_dir = os.getcwd()
proj_dir = os.path.abspath(os.path.join(curr_dir,'..','..')) ## use relative paths
data_dir = os.path.abspath(os.path.join(curr_dir,'..','..','data')) ## use relative paths 'D:\\data'
results_dir = os.path.join(proj_dir, 'csv')

###############################################################################################
################### GENERAL HELPERS ###########################################################
###############################################################################################

#### Helper data loader functions
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

def bootstrapCI(x,nIter=1000):
    '''
    input: x is an array
    '''
    u = []
    for i in np.arange(nIter):
        inds = np.random.RandomState(i).choice(len(x),len(x))
        boot = x[inds]
        u.append(np.mean(boot))

    p1 = len([i for i in u if i<0])/len(u) * 2 ## first version of p-value reflects number of samples that have value below 0
    p2 = len([i for i in u if i>0])/len(u) * 2 ## second version of p-value reflects number of samples that have value above 0
    U = np.mean(u)
    lb = np.percentile(u,2.5)
    ub = np.percentile(u,97.5)
    return U,lb,ub,p1,p2

###############################################################################################
################### SPECIFIC HELPERS FOR JUPYTER NOTEBOOKS ####################################
###############################################################################################


def make_drawing_predictions(sub_list,roi_list,version='4way',logged=True):
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
        print('Now analyzing {} ...'.format(this_roi))
        clear_output(wait=True)
        acc = []
        for this_sub in sub_list:
            ## load subject data in
            DM, DF = load_draw_data(this_sub,this_roi)
            try:
                RM12, RF12 = load_recog_data(this_sub,this_roi,'12')
            except:
                that_roi = draw_to_recog_roi_dict[this_roi]
                RM12, RF12 = load_recog_data(this_sub,that_roi,'12')
            #RM34, RF34 = load_recog_data(this_sub,this_roi,'34')
            #RM = pd.concat([RM12,RM34])
            #RF = np.vstack((RF12,RF34))
            RM = RM12
            RF = RF12
            
            assert RF.shape[1]==DF.shape[1] ## that number of voxels is identical

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
                    _DF = normalize(DF)
                else:
                    _RF = RF
                    _DF = DF

                # single train/test split
                X_train = _RF
                y_train = RM.label.values

                X_test = _DF
                y_test = DM.label.values
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

    return ALLDM, Acc


def make_prepostrecog_predictions(sub_list,roi_list,version='4way',test_phase='pre',logged=True):
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
                print 'Invalid test split, test_phase should be either "pre" or "post." '
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
        saves PNG versions of plots in plots dir, which is located at top level of project directory
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
        if not os.path.exists(os.path.join(proj_dir,'plots/{}/{}/{}'.format(nb_name,lookup[this_iv],toop))):
            os.makedirs(os.path.join(proj_dir,'plots/{}/{}/{}'.format(nb_name,lookup[this_iv],toop)))
        plt.tight_layout(rect=[0,0,1,0.7])
        plt.savefig(os.path.join(proj_dir,'plots/{}/{}/{}/prob_timecourse_{}_by_{}_{}.pdf'.\
                    format(nb_name,lookup[this_iv],toop,this_roi,lookup[this_iv],version)))
        plt.close(fig)    


def add_target_prob_column(df):
    '''
    df is dataframe, e.g., ALLPRE or ALLPOST that contains classifier probabilities for recognition runs
    in either the pre or post phases, respectively.

    '''
    
    df['target_prob_raw'] = np.nan
    df['trained'] = np.bool

    for ind,d in df.iterrows():
        print 'Analyzing {} of {}'.format(ind,df.shape[0])
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


        
