from __future__ import division
import pymongo as pm
import numpy as np
import tabular as tb
import json
import re
import cPickle
import copy
import time
import os
import glob

from pylab import *
from numpy import *

conn = pm.MongoClient(port=20809)
DBNAME = conn['during_morph_drawing_recognition']
COLNAME = DBNAME['fmri3.files']
coll=COLNAME
DATADIR = 'regressors'

mdtd = cPickle.load(open('neurosketch_design.pkl'))

# patient ID and worker ID mappings
## exceptions: '1115161_neurosketch', '1116161_neurosketch', '1117161_neurosketch', '1207161_neurosketch'
patient_ids = ['1121161_neurosketch', '1130161_neurosketch', 
'1201161_neurosketch', '1202161_neurosketch', '1203161_neurosketch',
'1206161_neurosketch', '1206162_neurosketch','1206163_neurosketch',
'1207161_neurosketch','1207162_neurosketch']
patient_ids = ['1207162_neurosketch']
patient_ids = ['0110171_neurosketch', '0110172_neurosketch',
'0111171_neurosketch','0112171_neurosketch', '0112172_neurosketch','0112173_neurosketch',
'0113171_neurosketch','0115172_neurosketch','0115174_neurosketch','0117171_neurosketch',
'0118171_neurosketch','0118172_neurosketch','0119171_neurosketch','0119172_neurosketch',
'0119173_neurosketch', '0119174_neurosketch','0120171_neurosketch','0120172_neurosketch',
'0120173_neurosketch','0123171_neurosketch','0123172_neurosketch','0123173_neurosketch',
'0124171_neurosketch','0125171_neurosketch','0125172_neurosketch']

all_wIDs = patient_ids
wID_pIDs = zip(patient_ids,all_wIDs)

def get_pID_from_wID(wID):
	pID = [i[0] for i in wID_pIDs if i[1] == wID]
	return pID[0]

def get_phase_index(x):
    if x=='pre':
        return 1
    elif x=='training':
        return 2
    elif x=='post':
        return 3

def get_category_index(x):
    if x=='car':
        return 1
    elif x=='furniture':
        return 2

def get_object_index(morphline,morphnum):
    furniture_axes = ['bedChair', 'bedTable', 'benchBed', 'chairBench', 'chairTable', 'tableBench']
    car_axes = ['limoToSUV','limoToSedan','limoToSmart','smartToSedan','suvToSedan','suvToSmart']  
    furniture_items = ['bed','bench','chair','table']
    car_items = ['limo','sedan','smartcar','SUV']               
    endpoints = getEndpoints(morphline)
    morphnum = float(morphnum)
    whichEndpoint = int(np.round(morphnum/100))
    thing = endpoints[whichEndpoint]
    if morphline in furniture_axes:
        return furniture_items.index(thing)+1
    elif morphline in car_axes:
        return car_items.index(thing)+1

def get_category_name(x):
    if x==1:
        return 'car'
    elif x==2:
        return 'furniture'

def get_object_name(cat_ind,obj_ind):
	furniture_items = ['bed','bench','chair','table']
	car_items = ['limo','sedan','smartcar','SUV']  	
	if cat_ind == 1: # car
		return car_items[obj_ind-1]
	elif cat_ind == 2: # furniture
		return furniture_items[obj_ind-1]

def get_response_index(x):
    furniture_items = ['bed','bench','chair','table']
    car_items = ['limo','sedan','smartcar','SUV']    
    if x in furniture_items:
        return furniture_items.index(x)
    elif x in car_items:
        return car_items.index(x)

def get_sequence_test_worker_list():
	wIDs = COLNAME.find({'wID':{'$ne': ''}}).distinct('wID')
	good_sessions = []
	funky_sessions = []
	good_sessions.append('1020161_neurosketch')
	return good_sessions, funky_sessions

def get_recog_pilot_worker_list():
	wIDs = COLNAME.find({'wID':{'$ne': ''}}).distinct('wID')
	good_sessions = []
	funky_sessions = []
	good_sessions = patient_ids
	return good_sessions, funky_sessions

def get_worker_list():
	wIDs = COLNAME.find({'wID':{'$ne': ''}}).distinct('wID')
	good_sessions = patient_ids
	return good_sessions	


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

def get_meta(w):

	import time
	start = time.time()

	# session parameters
	numPreRuns = 4
	numTrainRuns = 4
	numPostRuns = 2
	numTrialsTrainRun = 10
	numTrialsPreRun = 80
	numTrialsPreTotal = numTrialsPreRun*numPreRuns
	numTrialsTrainTotal = numTrialsTrainRun*numTrainRuns
	numTrials = numTrialsPreTotal*2 + numTrialsTrainTotal

	wID = []
	version = []
	category = []
	obj = []
	option1 = []
	option2 = []
	response = []
	viewpoint = []
	trialDuration = []
	phase = []
	trial = []
	counter = 0
	session_duration = []
	workers_complete = []
	TRnum = [] ## which TR of this block?
	runNum = [] ## function run index
	condition = []
	stim_onset = []

	try:
	    these = coll.find({'wID': w}).sort('trialNum')   
	    print these.count()
	    versionNum = these[0]['versionNum']
	    design = [i for i in mdtd if i['version'] == int(versionNum)] # find which axes belong to which condition
	    trained = design[0]['trained']
	    near = design[0]['near']
	    far1 = design[0]['far1']
	    far2 = design[0]['far2']

	    if these.count() > 9: 
	        a = coll.find({'wID':w}).distinct('trialNum')
	        upload_times = sort(map(math.trunc,map(float,a)))
	        workers_complete.append(w)
	        session_duration.append((max(upload_times)-min(upload_times))/1000/60)        
	        counter = 0
	        for ut in upload_times:
	            ##aa = coll.find({'wID':w, 'trialNum': {'$in': [re.compile('.*' + str(ut) + '.*')]}}).sort('trialNum')
	            aa = coll.find({'wID':w, 'trialNum': str(ut)}).sort('uploadDate')            
	            ####=========######=======########===========##########==========##############
	            if (aa[0]['task'] == 'recognition'):
	                # print ut                
	                assert 'cumulative_data' in aa[0].keys()
	                cumulData = aa[0]['cumulative_data']
	                trialdata = json.loads(cumulData)
	                if ut==319:
	                    print len(trialdata)
	                for rec in trialdata:                    
	                    if (rec['task']=='recognition') & (rec['trialNum'] not in trial): ## this means it is a full block and it is the recognition task
	                        wID.append(rec['wID'])
	                        version.append(int(rec['versionNum']))
	                        category.append(rec['category'])
	                        obj.append(get_object_name(2,get_object_index(rec['morphline'],rec['morphnum'])))
	                        viewpoint.append(rec['viewpoint'])
	                        phase.append(get_phase_index(rec['phase']))
	                        trial.append(int(rec['trialNum'])) 
	                        TRnum.append(int(np.round((rec['stimOnset']-rec['startBlock'])/1000/1.5)))
	                        stim_onset.append(float(rec['stimOnset'])-float(rec['startBlock']))
	                        condition.append([trained,near,far1,far2].index(rec['morphline'])) # 0=Trained, 1=Near, 2=Far1, 3=Far2
	                        if float(aa[0]['trialNum'])==0:
	                            runNum.append(1)
	                        elif float(aa[0]['trialNum']) <=numTrialsPreTotal: ## pre/localizer
	                            runNum.append(ceil(float(rec['trialNum']+1)/numTrialsPreRun))
	                        elif float(aa[0]['trialNum']) >= numTrialsPreTotal+numTrialsTrainTotal: ## post
	                            runNum.append(ceil((float(aa[0]['trialNum'])-40+1)/numTrialsPreRun)+numTrainRuns) 
	                        elif (float(aa[0]['trialNum'])>numTrialsPreTotal) & (float(aa[0]['trialNum'])< (numTrialsPreTotal+numTrialsTrainTotal)): ## train
	                            runNum.append(ceil((float(aa[0]['trialNum'])-numTrialsPreTotal)/numTrialsTrainRun)+numPreRuns)                                                   
	                        try:
	                            assert len(rec['options'])==2   
	                            response.append(rec['response'])
	                        except AssertionError:                         
	                            response.append('')                            
	                            pass
	                if int(aa[0]['trialNum']) not in trial:
	                    ## add last trial's data from this round
	                    wID.append(aa[0]['wID'])
	                    version.append(int(aa[0]['versionNum']))
	                    category.append(get_category_index(aa[0]['category']))
	                    obj.append(get_object_name(2,get_object_index(aa[0]['morphline'],aa[0]['morphnum'])))
	                    viewpoint.append(aa[0]['viewpoint'])
	                    phase.append(get_phase_index(aa[0]['phase']))
	                    trial.append(int(aa[0]['trialNum'])) 
	                    TRnum.append(int(np.round((float(aa[0]['stimOnset'])-float(aa[0]['startBlock']))/1000/1.5)))  
	                    stim_onset.append(float(aa[0]['stimOnset'])-float(aa[0]['startBlock']))
	                    if float(aa[0]['trialNum'])==0:
	                        runNum.append(1)
	                    elif float(aa[0]['trialNum']) <=numTrialsPreTotal:
	                        runNum.append(ceil(float(rec['trialNum']+1)/numTrialsPreRun))
	                    elif float(aa[0]['trialNum']) >= numTrialsPreTotal+numTrialsTrainTotal:
	                        runNum.append(ceil((float(aa[0]['trialNum'])-40+1)/numTrialsPreRun)+numTrainRuns) 
	                    elif (float(aa[0]['trialNum'])>numTrialsPreTotal) & (float(aa[0]['trialNum'])< (numTrialsPreTotal+numTrialsTrainTotal)):
	                        runNum.append(ceil((float(aa[0]['trialNum'])-numTrialsPreTotal)/numTrialsTrainRun)+numPreRuns)                         
	                    options = getEndpoints(aa[0]['morphline'])  
	                    response.append(aa[0]['response'])
	                    condition.append([trained,near,far1,far2].index(aa[0]['morphline'])) # 0=Trained, 1=Near, 2=Far1, 3=Far2

	except AssertionError,e:
	    print str(e)
	    pass

	X = tb.tabarray(columns = [map(str,wID),map(int,version),map(str,category),map(str,obj),map(int,trial), \
	                           TRnum,condition,np.round(stim_onset,2), map(int,runNum), map(int,viewpoint),map(str,response)], 
	                names = ['wID','versionNum','category','object', 'trial', \
	                         'TRnum','condition','onset_time', 'run_num','viewpoint','response']) 

	end = time.time()
	elapsed = end - start
	print "Time taken: ", elapsed/60, "minutes."

	## unroll from trial-wise meta into TR-wise meta
	import copy
	morphrecog = copy.deepcopy(X)
	TR_unfurled = np.arange(1,max(TRnum))

	onset = np.zeros(len(morphrecog)).astype(int)
	morphrecog = morphrecog.addcols(onset,names='onset')

	meta = []
	counter = 0
	numTrials =len(morphrecog)
	runStartDelayTR = 8 # number of TRs delay before first stim shown in each run


	for counter in np.arange(numTrials):
	    onset = morphrecog[counter]['TRnum']
	    if onset != 8: ## NOT the first stim of a given run
	        if counter < numTrials-1:        
	            upcoming = morphrecog[counter+1]['TRnum'] # how many TR's until the next stim, repeat that many times
	        else:
	            upcoming = 240
	        soa = upcoming-onset
	        if soa>0: # soa<0 after transition to new block
	            a = np.tile(morphrecog[counter],soa) 
	            a['TRnum']=np.arange(a['TRnum'][0],a['TRnum'][0]+soa)
	        else:
	            a = np.tile(morphrecog[counter],240-onset) 
	            a['TRnum']=np.arange(a['TRnum'][0],a['TRnum'][0]+240-onset)                
	        a['onset'][0] = 1                    
	        if len(meta)==0:
	            meta = a
	        else:
	            meta = np.hstack((meta,a)) 
	    else:    ## the first stim of a given run   
	        # tack on first 12 TR's at the top of meta, where no stim shown
	        start = morphrecog[counter]['TRnum']
	        a = np.tile(morphrecog[counter],start) 
	        a['TRnum'] = np.arange(a['TRnum'][0])
	        a['condition'] = '-1'
	        a['onset_time'] = 0
	        if len(meta)==0:
	            meta = a
	        else:
	            meta = np.hstack((meta,a))

	        if counter < numTrials-1:        
	            upcoming = morphrecog[counter+1]['TRnum'] # how many TR's until the next stim, repeat that many times
	        else:
	            upcoming = 240
	        soa = upcoming-onset
	        if soa>0: # soa<0 after transition to new block
	            a = np.tile(morphrecog[counter],soa) 
	            a['TRnum']=np.arange(a['TRnum'][0],a['TRnum'][0]+soa)
	        else:
	            a = np.tile(morphrecog[counter],240-onset) 
	            a['TRnum']=np.arange(a['TRnum'][0],a['TRnum'][0]+240-onset)                
	        a['onset'][0] = 1                    
	        if len(meta)==0:
	            meta = a
	        else:
	            meta = np.hstack((meta,a))             

	meta2 = copy.deepcopy(meta)
	meta3 = []
	for m in meta2:    
	    meta3.append(list(m))
	meta3 = np.array(meta3)
	meta4 = tb.tabarray(meta3,names = ['wID','versionNum','category','object', 'trial', \
	                         'TRnum','condition','onset_time','run_num','viewpoint','response','onset'])


	return X,meta4	 ## X = trial-wise-meta, meta = TR-wise-meta            

def save_meta_to_csv(meta,w):
	meta.saveSV(os.path.join(DATADIR, w + '.csv'))

def get_meta_all_subs():
	workers = get_worker_list()
	for w in workers:
		print w
		X,meta = get_meta(w)
		save_meta_to_csv(meta,w)

def gen_regressor(DATADIR):    
	##this creates the 3-column text file that FEAT expects in order to construct the GLM
	##col 1: onset time in ms
	##col 2: duration in ms
	##col 3: 'scale' factor, so just put 1's here unless you have a really good reason to do something else
	##num_files = len(glob.glob(os.path.join('neurosketch_metadata/glmreg/','*.txt')))    
	datadir = os.path.join(os.getcwd(),DATADIR)
	files = glob.glob(datadir+'/*.csv')
	for f in files:
	    f0 = f.split('/')[-1] # just the actual file name itself, without full path to it
	    f00 = f0.split('.')[0] # just file name without extension
	    X0 = tb.tabarray(SVfile=f,verbosity=0)
	    inds = (X0['trial'] != -1) # filter for non-disdaq TR's
	    X1 = X0[inds]
	    objs = np.unique(X1['object'])
	    cats = np.unique(X1['category'])
	    conds = np.unique(X1['condition'])
	    runs = np.unique(X1['run_num'])
	    dur = 1000
	    for r in runs:
	        for o in objs:
	            obj_name = o
	            if r==9: # post runs change run number to "5" and "6"
	                rr = 5
	            elif r==10:
	                rr = 6 
	            else:
	                rr = r
	            print r, obj_name            
	            inds = (X1['object']==o) & (X1['run_num']==r) & (X1['condition'] != -1) & (X1['onset']==1)
	            X2 = X1[inds]
	            onset_times = X2['onset_time']
	            uniq_onset_times = np.unique(onset_times)/1000 # in seconds
	            durations = np.tile(dur,len(uniq_onset_times))/1000 # in seconds
	            scale = np.tile(1,len(uniq_onset_times))
	            X3 = tb.tabarray(columns=[uniq_onset_times,durations,scale],
	                            names = ['onset_time','duration','scale'],verbosity=0)
	            if os.path.exists(os.path.join(DATADIR, f00,'run_'+str(rr)))==False:
	                os.makedirs(os.path.join(DATADIR, f00,'run_'+str(rr))) 
	            np.savetxt(os.path.join(DATADIR, f00, 'run_'+str(rr), obj_name+'.txt'),X3,fmt='%.2f')
	    print 'Finished generating glm regressors for Feat.'





