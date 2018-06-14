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
import seaborn as sns
sns.set_context('poster')
colors = sns.color_palette("cubehelix", 5)

###############################################################################################
################### HELPERS FOR predict_obj_during_drawing_from_recog_runs notebook ###########
###############################################################################################

### globals
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

# z-score normalization to de-mean & standardize variances within-voxel
def normalize(X):
    X = X - X.mean(0)
    X = X / np.maximum(X.std(0), 1e-5)
    return X

def bootstrapCI(x,nIter):
    '''
    input: x is an array
    '''
    u = []
    for i in np.arange(nIter):
        inds = np.random.RandomState(i).choice(len(x),len(x))
        boot = x[inds]
        u.append(np.mean(boot))

    p1 = len([i for i in u if i<0])/len(u) * 2
    p2 = len([i for i in u if i>0])/len(u) * 2
    p = np.min([p1,p2])
    U = np.mean(u)
    lb = np.percentile(u,2.5)
    ub = np.percentile(u,97.5)
    return U,lb,ub,p


def get_fn_applied_to_prob_timecourse(iv,DM,func=None): 
    '''
    Instead of getting the mean of target, foil, control for each iv, 
    you can also pass a function of target and foil to get_prob_timecourse (eg `lambda target, foil: target-foil`)
    and get the iv-timecourse for the output of that instead
    (this enables performing operations on target and foil data before means are taken for each iv)    
    '''
    trained_objs = np.unique(DM.label.values)
    control_objs = [i for i in ['bed','bench','chair','table'] if i not in trained_objs]    
    
    t1 = trained_objs[0]
    t2 = trained_objs[1]
    c1 = control_objs[0]
    c2 = control_objs[1]
    
    if func:
        return np.vstack((DM[DM.label==t1].groupby('trial_num').apply(lambda x: func(x['t1_prob'], x['t2_prob'])).values,
                          DM[DM.label==t2].groupby('trial_num').apply(lambda x: func(x['t2_prob'], x['t1_prob'])).values)).mean(0)
    else:
        target = np.vstack((DM[DM.label==t1].groupby(iv)['t1_prob'].mean().values,
                       DM[DM.label==t2].groupby(iv)['t2_prob'].mean().values)).mean(0) ## target timecourse
        foil = np.vstack((DM[DM.label==t1].groupby(iv)['t2_prob'].mean().values,
                       DM[DM.label==t2].groupby(iv)['t1_prob'].mean().values)).mean(0) ## foil timecourse
        control = np.vstack((DM[DM.label==t1].groupby(iv)['c1_prob'].mean().values,
                            DM[DM.label==t1].groupby(iv)['c2_prob'].mean().values,
                            DM[DM.label==t2].groupby(iv)['c1_prob'].mean().values,
                            DM[DM.label==t2].groupby(iv)['c2_prob'].mean().values)).mean(0) ## control timecourse    
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

def get_log_prob_timecourse(iv,DM,version='4way'):
    trained_objs = np.unique(DM.label.values)
    control_objs = [i for i in ['bed','bench','chair','table'] if i not in trained_objs]
    
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

def flatten(x):
    return [item for sublist in x for item in sublist]

def cleanup_df(df):
    surplus = [i for i in df.columns if 'Unnamed' in i]
    df = df.drop(surplus,axis=1)
    return df

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

    ALLDM = []
    ## loop through all subjects and rois
    Acc = []
    for this_roi in roi_list:
        print('Now analyzing {} ...'.format(this_roi))
        acc = []
        for this_sub in sub_list:
            ## load subject data in
            RM12, RF12 = load_recog_data(this_sub,this_roi,'12')
            #RM34, RF34 = load_recog_data(this_sub,this_roi,'34')
            #RM = pd.concat([RM12,RM34])
            #RF = np.vstack((RF12,RF34))
            RM = RM12
            RF = RF12
            DM, DF = load_draw_data(this_sub,this_roi)
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
        ## plot it
        sns.tsplot(data=x,
                  time=lookup[this_iv],
                  unit='sub',
                  condition='condition',
                  value='probability',
                  ci=95)
        if render_cond==1:
            plt.ylim(0,0.5)
            plt.axhline(0.25,linestyle=':',color='k')  
            plt.legend(bbox_to_anchor=(0.8, 1.01))  
            plt.title('Classifier evidence by condition in {}'.format(this_roi))

        else:
            plt.ylim(-0.3,0.3)
            plt.axhline(0,linestyle=':',color='k')  
            plt.legend(bbox_to_anchor=(0.7, 1.01))                        
            plt.title('Difference in classifier evidence by condition in {}'.format(this_roi))             
        plt.xticks(np.arange(np.max(x[lookup[this_iv]].values)+1))
        if not os.path.exists(os.path.join(proj_dir,'plots/{}/{}/{}'.format(nb_name,lookup[this_iv],toop))):
            os.makedirs(os.path.join(proj_dir,'plots/{}/{}/{}'.format(nb_name,lookup[this_iv],toop)))
        plt.tight_layout()        
        plt.savefig(os.path.join(proj_dir,'plots/{}/{}/{}/prob_timecourse_{}_by_{}_{}.png'.\
                    format(nb_name,lookup[this_iv],toop,this_roi,lookup[this_iv],version)))
        plt.close(fig)

        
def get_log_odds(ALLDM,
                 this_iv = 'trial_num',
                 roi_list = ['V1','V2'],
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
    
    for this_roi in roi_list_recog:

        T = []
        F = []
        C = []
        Sub = []
        for sub in subs:
            inds = (ALLDM['roi']==this_roi) & (ALLDM['subj']==sub) if this_roi != 'VGG' else (ALLDM['roi']==this_roi) & (ALLDM['subj']==sub) & (ALLDM['time_point'] == 23)
            t,f,c = get_log_prob_timecourse(this_iv,ALLDM[inds],version=version) if logged else get_prob_timecourse(this_iv,ALLDM[inds],version=version)
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
        x.to_csv(os.path.join(proj_dir, 'csv/difference_logprobs_{}_{}_{}.csv'.format(version,this_roi,this_iv)),index=False)

    ## make dataframe with subject-level difference scores
    d = pd.DataFrame([sub_tf,sub_tc,sub_fc,roi])
    d = d.transpose()
    d.columns = ['target-foil','target-control','foil-control','roi']
    d = d.astype({'target-foil':'float64','target-control':'float64','foil-control':'float64'})

    ## output target-foil ratios
    if logged==True:
        print(d.groupby('roi')['target-foil'].apply(lambda x: np.mean(np.exp(x))))
        d.to_csv(os.path.join(proj_dir, 'csv/difference_logprobs_{}.csv'.format(version)),index=False)
    else:
        print(d.groupby('roi')['target-foil'].mean())
        d.to_csv(os.path.join(proj_dir, 'csv/difference_rawprobs_{}.csv'.format(version)),index=False)
        
    return d
        
###############################################################################################
################### HELPERS FOR prepost RSA analyses ##########################################
###############################################################################################

mdtd = [{'near': 'suvToSmart', 'version': 0, 'far1': 'benchBed', 'trained': 'limoToSedan', 'far2': 'chairTable'}, {'near': 'suvToSmart', 'version': 1, 'far1': 'bedChair', 'trained': 'limoToSedan', 'far2': 'tableBench'}, {'near': 'suvToSmart', 'version': 2, 'far1': 'bedTable', 'trained': 'limoToSedan', 'far2': 'chairBench'}, {'near': 'suvToSedan', 'version': 3, 'far1': 'benchBed', 'trained': 'limoToSmart', 'far2': 'chairTable'}, {'near': 'suvToSedan', 'version': 4, 'far1': 'bedChair', 'trained': 'limoToSmart', 'far2': 'tableBench'}, {'near': 'suvToSedan', 'version': 5, 'far1': 'bedTable', 'trained': 'limoToSmart', 'far2': 'chairBench'}, {'near': 'smartToSedan', 'version': 6, 'far1': 'benchBed', 'trained': 'limoToSUV', 'far2': 'chairTable'}, {'near': 'smartToSedan', 'version': 7, 'far1': 'bedChair', 'trained': 'limoToSUV', 'far2': 'tableBench'}, {'near': 'smartToSedan', 'version': 8, 'far1': 'bedTable', 'trained': 'limoToSUV', 'far2': 'chairBench'}, {'near': 'limoToSUV', 'version': 9, 'far1': 'benchBed', 'trained': 'smartToSedan', 'far2': 'chairTable'}, {'near': 'limoToSUV', 'version': 10, 'far1': 'bedChair', 'trained': 'smartToSedan', 'far2': 'tableBench'}, {'near': 'limoToSUV', 'version': 11, 'far1': 'bedTable', 'trained': 'smartToSedan', 'far2': 'chairBench'}, {'near': 'limoToSmart', 'version': 12, 'far1': 'benchBed', 'trained': 'suvToSedan', 'far2': 'chairTable'}, {'near': 'limoToSmart', 'version': 13, 'far1': 'bedChair', 'trained': 'suvToSedan', 'far2': 'tableBench'}, {'near': 'limoToSmart', 'version': 14, 'far1': 'bedTable', 'trained': 'suvToSedan', 'far2': 'chairBench'}, {'near': 'limoToSedan', 'version': 15, 'far1': 'benchBed', 'trained': 'suvToSmart', 'far2': 'chairTable'}, {'near': 'limoToSedan', 'version': 16, 'far1': 'bedChair', 'trained': 'suvToSmart', 'far2': 'tableBench'}, {'near': 'limoToSedan', 'version': 17, 'far1': 'bedTable', 'trained': 'suvToSmart', 'far2': 'chairBench'}, {'near': 'chairTable', 'version': 18, 'far1': 'limoToSedan', 'trained': 'benchBed', 'far2': 'suvToSmart'}, {'near': 'chairTable', 'version': 19, 'far1': 'limoToSmart', 'trained': 'benchBed', 'far2': 'suvToSedan'}, {'near': 'chairTable', 'version': 20, 'far1': 'limoToSUV', 'trained': 'benchBed', 'far2': 'smartToSedan'}, {'near': 'tableBench', 'version': 21, 'far1': 'limoToSedan', 'trained': 'bedChair', 'far2': 'suvToSmart'}, {'near': 'tableBench', 'version': 22, 'far1': 'limoToSmart', 'trained': 'bedChair', 'far2': 'suvToSedan'}, {'near': 'tableBench', 'version': 23, 'far1': 'limoToSUV', 'trained': 'bedChair', 'far2': 'smartToSedan'}, {'near': 'chairBench', 'version': 24, 'far1': 'limoToSedan', 'trained': 'bedTable', 'far2': 'suvToSmart'}, {'near': 'chairBench', 'version': 25, 'far1': 'limoToSmart', 'trained': 'bedTable', 'far2': 'suvToSedan'}, {'near': 'chairBench', 'version': 26, 'far1': 'limoToSUV', 'trained': 'bedTable', 'far2': 'smartToSedan'}, {'near': 'bedTable', 'version': 27, 'far1': 'limoToSedan', 'trained': 'chairBench', 'far2': 'suvToSmart'}, {'near': 'bedTable', 'version': 28, 'far1': 'limoToSmart', 'trained': 'chairBench', 'far2': 'suvToSedan'}, {'near': 'bedTable', 'version': 29, 'far1': 'limoToSUV', 'trained': 'chairBench', 'far2': 'smartToSedan'}, {'near': 'bedChair', 'version': 30, 'far1': 'limoToSedan', 'trained': 'tableBench', 'far2': 'suvToSmart'}, {'near': 'bedChair', 'version': 31, 'far1': 'limoToSmart', 'trained': 'tableBench', 'far2': 'suvToSedan'}, {'near': 'bedChair', 'version': 32, 'far1': 'limoToSUV', 'trained': 'tableBench', 'far2': 'smartToSedan'}, {'near': 'benchBed', 'version': 33, 'far1': 'limoToSedan', 'trained': 'chairTable', 'far2': 'suvToSmart'}, {'near': 'benchBed', 'version': 34, 'far1': 'limoToSmart', 'trained': 'chairTable', 'far2': 'suvToSedan'}, {'near': 'benchBed', 'version': 35, 'far1': 'limoToSUV', 'trained': 'chairTable', 'far2': 'smartToSedan'}]

# behavioral data from database (versionNums index each subject in mdtd)
coll = {'0110171_neurosketch': '18', '0110172_neurosketch': '21', '0111171_neurosketch': '24', '0112171_neurosketch': '27', '0112172_neurosketch': '30', '0112173_neurosketch': '33', '0113171_neurosketch': '21', '0115172_neurosketch': '24', '0115174_neurosketch': '27', '0117171_neurosketch': '30', '0118171_neurosketch': '33', '0118172_neurosketch': '33', '0119171_neurosketch': '18', '0119172_neurosketch': '21', '0119173_neurosketch': '24', '0119174_neurosketch': '27', '0120171_neurosketch': '30', '0120172_neurosketch': '33', '0120173_neurosketch': '18', '0123171_neurosketch': '24', '0123173_neurosketch': '30', '0124171_neurosketch': '33', '0125171_neurosketch': '27', '0125172_neurosketch': '21', '1121161_neurosketch': '24', '1130161_neurosketch': '27', '1201161_neurosketch': '30', '1202161_neurosketch': '21', '1203161_neurosketch': '33', '1206161_neurosketch': '18', '1206162_neurosketch': '21', '1206163_neurosketch': '24', '1207162_neurosketch': '30'}

### globals
obj2cope = {'bed':1,'bench':2, 'chair':3,'table':4}
roi_dir = 'DATA/copes/roi'              # path to roi .nii.gz files
cope_dir = 'DATA/copes/recog/objectGLM' # path to hires copes

def get_object_index(morphline,morphnum):
    furniture_axes = ['bedChair', 'bedTable', 'benchBed', 'chairBench', 'chairTable', 'tableBench']
    car_axes = ['limoToSUV','limoToSedan','limoToSmart','smartToSedan','suvToSedan','suvToSmart']
    furniture_items = ['bed','bench','chair','table']
    car_items = ['limo','sedan','smartcar','SUV']
    endpoints = mdr_helpers.getEndpoints(morphline)
    morphnum = float(morphnum)
    whichEndpoint = int(np.round(morphnum/100))
    thing = endpoints[whichEndpoint]
    if morphline in furniture_axes:
        return furniture_items.index(thing)+1
    elif morphline in car_axes:
        return car_items.index(thing)+1

def getEndpoints(morphline):
    if morphline=='sedanMinivan':
        return ['sedan','minivan']
    elif morphline=='minivanSportscar':
        return ['minivan','sportscar']
    elif morphline=='sportscarSUV':
        return ['sportscar','SUV']
    elif morphline=='SUVMinivan':
        return ['SUV','minivan']
    elif morphline=='sportscarSedan':
        return ['sportscar','sedan']
    elif morphline=='sedanSUV':
        return ['sedan','SUV']
    elif morphline=='bedChair':
        return ['bed','chair']
    elif morphline=='bedTable':
        return ['bed','table']
    elif morphline=='benchBed':
        return ['bench','bed']
    elif morphline=='chairBench':
        return ['chair','bench']
    elif morphline=='chairTable':
        return ['chair','table']
    elif morphline=='tableBench':
        return ['table','bench']
    elif morphline=='limoToSUV':
        return ['limo','SUV']
    elif morphline=='limoToSedan':
        return ['sedan','limo']
    elif morphline=='limoToSmart':
        return ['limo','smartcar']
    elif morphline=='smartToSedan':
        return ['smartcar','sedan']
    elif morphline=='suvToSedan':
        return ['SUV','sedan']
    elif morphline=='suvToSmart':
        return ['SUV','smartcar']
    else:
        return ['A','B']

def triple_sum(X):
    return sum(sum(sum(X)))

def get_mask_array(mask_path):
    mask_img = image.load_img(mask_path)
    mask_data = mask_img.get_data()
    num_brain_voxels = sum(sum(sum(mask_data==1)))
    return mask_data, num_brain_voxels

def load_roi_mask(subj, roi):
    if roi in ['Frontal', 'Parietal', 'supraMarginal', 'Insula', 'postCentral', 'preCentral']:
        mask_path = os.path.join(roi_dir, subj, roi + '_draw.nii.gz')
    elif roi in ['V1', 'V2']:
        mask_path = os.path.join(roi_dir, subj, roi+'.nii.gz')
    else:
        mask_path = os.path.join(roi_dir, subj, roi + '_FS.nii.gz')
    mask_data, nv = get_mask_array(mask_path)
    return mask_data

def normalize(X):
    mn = X.mean(0)
    sd = X.std(0)
    X = X - mn
    X = X / np.maximum(sd, 1e-5)
    return X

def load_single_run_weights(subj,run_num,cope_num):
    nifti_path = cope_dir + '/' + subj[:7] + '_run' + str(run_num) + '_cope' + str(cope_num) + '_hires.nii.gz'
    fmri_img = image.load_img(nifti_path)
    fmri_data = fmri_img.get_data()
    return fmri_data

def apply_mask(data,mask):
    return data[mask==1]

def load_data_and_apply_mask(subj,run_num,roi,cope_num):
    mask = load_roi_mask(subj,roi)
    vol = load_single_run_weights(subj,run_num,cope_num)
    vec = apply_mask(vol,mask)
    return vec

def extract_obj_by_voxel_run_mat(this_sub,run_num, roi):
    cope1 = load_data_and_apply_mask(this_sub,run_num,roi,1)
    cope2 = load_data_and_apply_mask(this_sub,run_num,roi,2)
    cope3 = load_data_and_apply_mask(this_sub,run_num,roi,3)
    cope4 = load_data_and_apply_mask(this_sub,run_num,roi,4)
    return np.vstack((cope1,cope2,cope3,cope4))

def plot_phase_RSM(this_sub,roi,phase):
    '''
    e.g., plot_phase_RSM(this_sub,'fusiform','pre')
    '''
    if phase=='pre':
        mat1 = extract_obj_by_voxel_run_mat(this_sub,3,roi)
        mat2 = extract_obj_by_voxel_run_mat(this_sub,4,roi)
    elif phase=='post':
        mat1 = extract_obj_by_voxel_run_mat(this_sub,5,roi)
        mat2 = extract_obj_by_voxel_run_mat(this_sub,6,roi)
    stacked = np.vstack((mat1,mat2))
    plt.matshow(np.corrcoef(stacked))
    plt.colorbar()

def extract_condition_by_voxel_run_mat(this_sub,run_num, roi):
    versionNum = coll[this_sub]

    design = [i for i in mdtd if i['version'] == int(versionNum)] # which axes belong to which condition?
    trained = design[0]['trained']
    near = design[0]['near']
    far1 = design[0]['far1']
    far2 = design[0]['far2']

    Tep = getEndpoints(trained)
    Nep = getEndpoints(near)
    condorder = Tep + Nep

    slot1 = load_data_and_apply_mask(this_sub,run_num,roi,obj2cope[condorder[0]])
    slot2 = load_data_and_apply_mask(this_sub,run_num,roi,obj2cope[condorder[1]])
    slot3 = load_data_and_apply_mask(this_sub,run_num,roi,obj2cope[condorder[2]])
    slot4 = load_data_and_apply_mask(this_sub,run_num,roi,obj2cope[condorder[3]])
    return np.vstack((slot1,slot2,slot3,slot4))

def remove_nans(array):
    return array[~np.isnan(array)]

def rmse(a):
    return np.sqrt(np.mean(map(np.square,a)))

def betwitdist(a,b,ab):
    return ab/np.sqrt(0.5*(a**2+b**2))

def norm_hist(data,bins):
    weights = np.ones_like(data)/float(len(data))
    plt.hist(data, bins=bins, weights=weights)

def compare_btw_wit_obj_similarity_across_runs(this_sub,phase,roi):
    if phase=='pre':
        mat1 = extract_obj_by_voxel_run_mat(this_sub,3,roi)
        mat2 = extract_obj_by_voxel_run_mat(this_sub,4,roi)
    elif phase=='post':
        mat1 = extract_obj_by_voxel_run_mat(this_sub,5,roi)
        mat2 = extract_obj_by_voxel_run_mat(this_sub,6,roi)
    fAB = np.vstack((mat1,mat2)) # stack feature matrices
    DAB = pairwise_distances(fAB, metric='correlation') # square matrix, where off-diagblock is distances *between* fA and fB vectors
    offblock = DAB[:len(mat1),range(len(mat1),shape(DAB)[1])]
    wit_obj = DAB[:len(mat1),range(len(mat1),shape(DAB)[1])].diagonal()
    btw_obj = np.hstack((offblock[np.triu_indices(shape(offblock)[0],k=1)],offblock[np.tril_indices(shape(offblock)[0],k=-1)]))
    wit_mean = wit_obj.mean()
    btw_mean = btw_obj.mean()
    return wit_mean,btw_mean

def compare_btw_wit_cond_similarity_across_runs(this_sub,phase,roi):
    if phase == 'pre':
        mat1 = extract_condition_by_voxel_run_mat(this_sub,3,roi)
        mat2 = extract_condition_by_voxel_run_mat(this_sub,4,roi)
    elif phase == 'post':
        mat1 = extract_condition_by_voxel_run_mat(this_sub,5,roi)
        mat2 = extract_condition_by_voxel_run_mat(this_sub,6,roi)
    elif phase == '35':
        mat1 = extract_condition_by_voxel_run_mat(this_sub,3,roi)
        mat2 = extract_condition_by_voxel_run_mat(this_sub,5,roi)
    elif phase == '46':
        mat1 = extract_condition_by_voxel_run_mat(this_sub,4,roi)
        mat2 = extract_condition_by_voxel_run_mat(this_sub,6,roi)

    fAB = np.vstack((mat1,mat2)) # stack feature matrices
    DAB = pairwise_distances(fAB, metric='correlation') # square matrix, where off-diagblock is distances *between* fA and fB vectors
    offblock = DAB[:len(mat1),range(len(mat1),shape(DAB)[1])]

    trained_witobj = offblock.diagonal()[:2]
    control_witobj = offblock.diagonal()[2:]
    trained_btwobj = np.array([offblock[:2,:2][0,1], offblock[:2,:2][1,0]])
    control_btwobj = np.array([offblock[2:,2:][0,1],offblock[2:,2:][1,0]])

    trawit_mean = trained_witobj.mean()
    conwit_mean = control_witobj.mean()
    trabtw_mean = trained_btwobj.mean()
    conbtw_mean = control_btwobj.mean()
    return trawit_mean,conwit_mean,trabtw_mean,conbtw_mean

def get_vectorized_voxels_from_map(filename):
    img = nib.load(filename)
    data = img.get_data()
    flat = np.ravel(data)
    return flat



###############################################################################################
################### HELPERS FOR relate drawing and prepost ####################################
###############################################################################################

def corrbootstrapCI(x, y, nIter):
    '''
    input:
        x is an array
        y is an array
        nIter is the numberof random samples to take
    returns:
        U: bootstrapped mean
        lb: lower bound of 95 CI
        ub: upper bound of 95 CI
        p: p value
    '''
    u = []
    for i in np.arange(nIter):
        inds = np.random.RandomState(i).choice(len(x),len(x))
        bootx = x[inds]
        booty = y[inds]
        _corr = stats.pearsonr(bootx, booty)[0]
        corr = pd.DataFrame([bootx, booty]).transpose().corr()[0][1] if np.isnan(_corr) else _corr
        u.append(corr)

    p1 = len([i for i in u if i<0])/len(u) * 2
    p2 = len([i for i in u if i>0])/len(u) * 2
    p = np.min([p1,p2])
    U = np.mean(u)
    lb = np.percentile(u,2.5)
    ub = np.percentile(u,97.5)
    return U,lb,ub,p


def custom_bootstrapCI(x, estimator, nIter, *args):
    u = []
    for i in np.arange(nIter):
        inds = np.random.RandomState(i).choice(len(x), len(x))
        boot = x[inds]
        u.append(estimator(boot, *args))

    p1 = len([i for i in u if i < 0]) / len(u) * 2
    p2 = len([i for i in u if i > 0]) / len(u) * 2
    p = np.min([p1, p2])
    U = np.mean(u)
    lb = np.percentile(u, 2.5)
    ub = np.percentile(u, 97.5)
    return U, lb, ub, p


def compute_clf_measure(target, foil, measure):
    if measure == 't-f':
        return target - foil
    elif measure == 'txf':
        return target + foil if logged else target * foil
    elif measure == 't':
        return target
    else:
        return foil


def scoreVSdiff(subdata, this_roi):
    clfscores = [np.mean(c['clf']) for c in subdata]
    diffscores = [sub['diff'] for sub in subdata]

    if this_roi == 'Frontal':
        return pd.DataFrame([clfscores, diffscores]).transpose().corr()[0][1]
    else:
        return stats.pearsonr(clfscores, diffscores)[0]


def slope_scoreVSdiff(subdata, this_roi, num_ivs):
    diffscores = [sub['diff'] for sub in subdata]
    clfscores = [c['clf'] for c in subdata]

    if this_roi == 'Frontal':
        coefficients = [pd.DataFrame([[c[i] for c in clfscores], diffscores]).transpose().corr()[0][1] for i in
                        range(num_ivs)]
    else:
        coefficients = [stats.pearsonr([c[i] for c in clfscores], diffscores)[0] for i in range(num_ivs)]
    return linregress(np.arange(num_ivs), coefficients)[0]
