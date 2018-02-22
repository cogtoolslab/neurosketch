import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import linear_model, datasets, neighbors
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn import svm
import seaborn as sns

###############################################################################################
################### HELPERS FOR predict_obj_during_drawing_from_recog_runs notebook ###########
###############################################################################################

### globals

## define path to input datasets (tidy format)
path_to_recog = '/home/jefan/neurosketch/neurosketch_voxelmat_freesurfer_recog'
path_to_draw = '/home/jefan/neurosketch/neurosketch_voxelmat_freesurfer_drawing'
roi_list = np.array(['V1','V2','LOC','IT','fusiform','parahippo', 'PRC', 'ento','hipp','mOFC'])

## general plotting params
sns.set_context('poster')
colors = sns.color_palette("cubehelix", 5)

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

## plotting helper
def get_prob_timecourse(iv,DM,version='4way'):
    trained_objs = np.unique(DM.label.values)
    control_objs = [i for i in ['bed','bench','chair','table'] if i not in trained_objs]    
    
    if version=='4way':
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
    elif version=='3way':
        t1 = trained_objs[0]
        t2 = trained_objs[1]
        target = np.vstack((DM[DM.label==t1].groupby(iv)['t1_prob'].mean().values,
                       DM[DM.label==t2].groupby(iv)['t2_prob'].mean().values)).mean(0) ## target timecourse; mean is taken over what?
        foil = np.vstack((DM[DM.label==t1].groupby(iv)['t2_prob'].mean().values,
                       DM[DM.label==t2].groupby(iv)['t1_prob'].mean().values)).mean(0) ## foil timecourse
        control = np.vstack((DM[DM.label==t1].groupby(iv)['c_prob'].mean().values,
                            DM[DM.label==t2].groupby(iv)['c_prob'].mean().values)).mean(0) ## control timecourse
        
    elif version=='2way':
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
            3way: subsamples one of the control objects, trains 3-way classifier
                    that outputs probabilities for target, foil, and control objects
                    that is then aggregated across classifiers
            2way: trains to discriminate only the two trained objects from recognition runs
                    then makes predictions on drawing data
        logged: boolean. If true, return log-probabilities. If false, return raw probabilities.
                    
    assumes: that you have directories containing recognition run and drawing run data, consisting of paired .npy 
                voxel matrices and .csv metadata matrices
    '''

    ALLDM = []
    ## loop through all subjects and rois
    Acc = []
    for this_roi in roi_list:
        print(this_roi)
        acc = []
        for this_sub in sub_list:
            ## load subject data in
            RM12, RF12 = load_recog_data(this_sub,this_roi,'12')
            RM34, RF34 = load_recog_data(this_sub,this_roi,'34')
            RM = pd.concat([RM12,RM34])
            RF = np.vstack((RF12,RF34))
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

            DM['subj'] = np.repeat(this_sub,DM.shape[0])
            DM['roi'] = np.repeat(this_roi,DM.shape[0])

            if len(ALLDM)==0:
                ALLDM = DM
            else:
                ALLDM = pd.concat([ALLDM,DM],ignore_index=True)

            acc.append(clf.score(X_test, y_test))
            
            
            '''
            ## plot probability timecourse by run number
            fig = plt.figure(figsize=(5,5))
            iv = 'run_num'
            t,f,c = get_prob_timecourse(iv,DM)
            plt.plot(t,color=colors[0],label='target')
            plt.plot(f,color=colors[1],label='foil')
            plt.plot(c,color=colors[2],label='control')
            plt.legend(bbox_to_anchor=(1.45, 1.01))
            plt.ylim(0,1)
            plt.xlabel(iv)
            plt.ylabel('probability')
            if not os.path.exists('./plots/subj'):
                os.makedirs('./plots/subj')
            plt.tight_layout()
            plt.savefig('./plots/subj/{}_{}_prob_{}.pdf'.format(iv.split('_')[0],this_roi,this_sub))
            plt.close(fig)  
            '''                        

        Acc.append(acc)

    return ALLDM, Acc


## plotting helper
def get_prob_timecourse(iv,DM,version='4way'):
    trained_objs = np.unique(DM.label.values)
    control_objs = [i for i in ['bed','bench','chair','table'] if i not in trained_objs]    
    
    if version=='4way':
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
    elif version=='3way':
        t1 = trained_objs[0]
        t2 = trained_objs[1]
        target = np.vstack((DM[DM.label==t1].groupby(iv)['t1_prob'].mean().values,
                       DM[DM.label==t2].groupby(iv)['t2_prob'].mean().values)).mean(0) ## target timecourse
        foil = np.vstack((DM[DM.label==t1].groupby(iv)['t2_prob'].mean().values,
                       DM[DM.label==t2].groupby(iv)['t1_prob'].mean().values)).mean(0) ## foil timecourse
        control = np.vstack((DM[DM.label==t1].groupby(iv)['c_prob'].mean().values,
                            DM[DM.label==t2].groupby(iv)['c_prob'].mean().values)).mean(0) ## control timecourse
        
    elif version=='2way':
        t1 = trained_objs[0]
        t2 = trained_objs[1]
        target = np.vstack((DM[DM.label==t1].groupby(iv)['t1_prob'].mean().values,
                       DM[DM.label==t2].groupby(iv)['t2_prob'].mean().values)).mean(0) ## target timecourse; mean is taken over what?
        foil = np.vstack((DM[DM.label==t1].groupby(iv)['t2_prob'].mean().values,
                       DM[DM.label==t2].groupby(iv)['t1_prob'].mean().values)).mean(0) ## foil timecourse
        
        control = np.zeros(len(foil))        
        
    return target, foil, control


###############################################################################################
################### HELPERS FOR prepost RSA analyses ##########################################
###############################################################################################

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
    
def load_roi_mask(subj,run_num,roi):
    mask_path = proj_dir + subj +'/analysis/firstlevel/rois/' + roi + '_func__' + str(run_num) + '_binarized.nii.gz'        
    mask_data, nv = get_mask_array(mask_path)
    return mask_data

def load_roi_mask_combined(subj,run_num,roi):
    if run_num in [1,2]:
        phase_num = '12'
    elif run_num in [3,4]:
        phase_num = '34'
    elif run_num in [5,6]:
        phase_num = '56'
    mask_path = proj_dir + '/' + subj +'/analysis/firstlevel/rois/' + roi + '_func_combined_' + phase_num + '_binarized.nii.gz'        
    mask_data, nv = get_mask_array(mask_path)
    return mask_data

def normalize(X):
    mn = X.mean(0)
    sd = X.std(0)
    X = X - mn
    X = X / np.maximum(sd, 1e-5)
    return X

def load_single_run_weights(subj,run_num,cope_num):
    nifti_path = proj_dir + '/' + subj + '/analysis/firstlevel/glm4_recognition_run_' + str(run_num) + \
                '.feat/stats/' + 'cope' + str(cope_num) + '.nii.gz'
    fmri_img = image.load_img(nifti_path)
    fmri_data = fmri_img.get_data()
    return fmri_data

def apply_mask(data,mask):
    return data[mask==1]

def load_data_and_apply_mask(subj,run_num,roi,cope_num):
    mask = load_roi_mask_combined(subj,run_num,roi)
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
    w = this_sub
    these = coll.find({'wID': w}).sort('trialNum')   
    versionNum = these[0]['versionNum']

    design = [i for i in mdtd if i['version'] == int(versionNum)] # find which axes belong to which condition
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
    DAB = sklearn.metrics.pairwise.pairwise_distances(fAB, metric='correlation') # square matrix, where off-diagblock is distances *between* fA and fB vectors
    offblock = DAB[:len(mat1),range(len(mat1),shape(DAB)[1])]
    wit_obj = DAB[:len(mat1),range(len(mat1),shape(DAB)[1])].diagonal()
    btw_obj = np.hstack((offblock[np.triu_indices(shape(offblock)[0],k=1)],offblock[np.tril_indices(shape(offblock)[0],k=-1)]))
    wit_mean = wit_obj.mean()
    btw_mean = btw_obj.mean()
    return wit_mean,btw_mean

def compare_btw_wit_cond_similarity_across_runs(this_sub,phase,roi):

    if phase=='pre':
        mat1 = extract_condition_by_voxel_run_mat(this_sub,3,roi)
        mat2 = extract_condition_by_voxel_run_mat(this_sub,4,roi)
    elif phase=='post':
        mat1 = extract_condition_by_voxel_run_mat(this_sub,5,roi)
        mat2 = extract_condition_by_voxel_run_mat(this_sub,6,roi)

    fAB = np.vstack((mat1,mat2)) # stack feature matrices
    DAB = sklearn.metrics.pairwise.pairwise_distances(fAB, metric='correlation') # square matrix, where off-diagblock is distances *between* fA and fB vectors
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

