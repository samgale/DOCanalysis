# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 16:22:30 2022

@author: svc_ccg
"""

import copy
import math
import os
import pickle
import warnings
import numpy as np
import scipy.stats
import pandas as pd
import h5py
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
import sklearn
import fileIO
from DynamicRoutingAnalysisUtils import calcDprime


## make h5df with binned spike counts
from allensdk.brain_observatory.behavior.behavior_project_cache.behavior_neuropixels_project_cache import VisualBehaviorNeuropixelsProjectCache

def getSpikeBins(spikeTimes,startTimes,windowDur,binSize=0.001):
    bins = np.arange(0,windowDur+binSize,binSize)
    spikes = np.zeros((len(startTimes),bins.size-1),dtype=bool)
    for i,start in enumerate(startTimes):
        startInd = np.searchsorted(spikeTimes,start)
        endInd = np.searchsorted(spikeTimes,start+windowDur)
        spikes[i] = np.histogram(spikeTimes[startInd:endInd]-start, bins)[0]
    return spikes

vbnCache = r'\\allen\aibs\informatics\chris.morrison\ticket-27\allensdk_caches\vbn_cache_2022_Jul29'

cache = VisualBehaviorNeuropixelsProjectCache.from_local_cache(cache_dir=vbnCache)

sessions = cache.get_ecephys_session_table(filter_abnormalities=False)

windowDur = 0.75
binSize = 0.001
nBins = int(windowDur/binSize)

h5Path = r'C:/Users/svc_ccg/Desktop/Analysis/vbnAllUnitSpikeTensor.hdf5'
h5File = h5py.File(h5Path,'w')

sessionCount = 0
for sessionId,sessionData in sessions.iterrows():
    sessionCount += 1
    print('session '+str(sessionCount))
    
    session = cache.get_ecephys_session(ecephys_session_id=sessionId)
    
    stim = session.stimulus_presentations
    flashTimes = stim.start_time[stim.active]
    
    units = session.get_units()
    channels = session.get_channels()
    units = units.merge(channels,left_on='peak_channel_id',right_index=True)
    goodUnits = units[(units['quality']=='good') & (units['snr']>1) & (units['isi_violations']<1)]
    spikeTimes = session.spike_times
    
    h5Group = h5File.create_group(str(sessionId))
    h5Group.create_dataset('unitIds',data=goodUnits.index,compression='gzip',compression_opts=4)
    spikes = h5Group.create_dataset('spikes',shape=(len(goodUnits),len(flashTimes),nBins),dtype=bool,chunks=(1,len(flashTimes),nBins),compression='gzip',compression_opts=4)
    
    i = 0
    for unitId,unitData in goodUnits.iterrows(): 
        spikes[i] = getSpikeBins(spikeTimes[unitId],flashTimes,windowDur,binSize)
        i += 1

h5File.close()
    


##
baseDir = r"C:\Users\svc_ccg\Desktop\Analysis\vbn"

stimTable = pd.read_csv(os.path.join(baseDir,'master_stim_table.csv'))

unitTable = pd.read_csv(os.path.join(baseDir,'units_with_cortical_layers.csv'))

unitData = h5py.File(os.path.join(baseDir,'vbnAllUnitSpikeTensor.hdf5'),mode='r')

binSize = 0.001
baseWin = slice(680,750)
respWin = slice(30,100)


def findResponsiveUnits(basePsth,respPsth,baseWin,respWin):
    hasSpikes = ((respPsth[:,:,respWin].mean(axis=(1,2)) - basePsth[:,:,baseWin].mean(axis=(1,2))) / 0.001) > 0.1
    
    base = basePsth[:,:,baseWin].mean(axis=1)
    resp = respPsth[:,:,respWin].mean(axis=1)
    peak = np.max(resp-base.mean(axis=1)[:,None],axis=1)
    hasPeakResp = peak > 5 * base.std(axis=1)
    
    base = basePsth[:,:,baseWin].mean(axis=2)
    resp = respPsth[:,:,respWin].mean(axis=2)
    pval = np.array([1 if np.sum(r-b)==0 else scipy.stats.wilcoxon(b,r)[1] for b,r in zip(base,resp)])
    
    return hasSpikes & hasPeakResp & (pval<0.05)

    
## count responsive units
sessionIds = stimTable['session_id'].unique()

regions = np.unique(unitTable['structure_acronym'])

nUnits = {region: [] for region in regions}

for sessionIndex,sessionId in enumerate(sessionIds):
    print('session '+str(sessionIndex+1))
    units = unitTable.set_index('unit_id').loc[unitData[str(sessionId)]['unitIds'][:]]
    spikes = unitData[str(sessionId)]['spikes']
    
    stim = stimTable[(stimTable['session_id']==sessionId) & stimTable['active']].reset_index()
    autoRewarded = np.array(stim['auto_rewarded']).astype(bool)
    changeFlash = np.where(stim['is_change'] & ~autoRewarded)[0]
    
    for region in regions:
        print(region)
        inRegion = np.array(units['structure_acronym']==region)
        if any(inRegion):
            s = np.zeros((inRegion.sum(),spikes.shape[1],spikes.shape[2]),dtype=bool)
            for i,u in enumerate(np.where(inRegion)[0]):
                s[i]=spikes[u,:,:]
            changeSp = s[:,changeFlash,:]
            preChangeSp = s[:,changeFlash-1,:]
            hasResp = findResponsiveUnits(preChangeSp,changeSp,baseWin,respWin)
            nUnits[region].append(hasResp.sum())
            
for region in nUnits:
    print(region)
    print(sum([n>19 for n in nUnits[region]]))

# save result to pkl file
pkl = fileIO.saveFile(fileType='*.pkl')
pickle.dump(nUnits,open(pkl,'wb'))

# get result from pkl file
pkl = fileIO.getFile(fileType='*.pkl')
nUnits = pickle.load(open(pkl,'rb'))


## adaptation
sessionIds = stimTable['session_id'][stimTable['experience_level']=='Familiar'].unique()

regions = ('LGd','VISp','VISl','VISrl','VISal','VISpm','VISam','LP','MRN')
layers = (('1','2/3'),'4','5',('6a','6b'))

changeSpikes = {region: {layer: [] for layer in layers} for region in regions}
preChangeSpikes = copy.deepcopy(changeSpikes)  
changeResp = copy.deepcopy(changeSpikes)
preChangeResp = copy.deepcopy(changeSpikes)  
adaptSpikes = copy.deepcopy(changeSpikes)
adaptResp = copy.deepcopy(changeSpikes)
flashBase = copy.deepcopy(changeSpikes)
flashResp = copy.deepcopy(changeSpikes)
changeFlashBase = copy.deepcopy(changeSpikes)
changeFlashResp = copy.deepcopy(changeSpikes)
for sessionIndex,sessionId in enumerate(sessionIds):
    units = unitTable.set_index('unit_id').loc[unitData[str(sessionId)]['unitIds'][:]]
    spikes = unitData[str(sessionId)]['spikes']
    
    stim = stimTable[(stimTable['session_id']==sessionId) & stimTable['active']].reset_index()
    autoRewarded = np.array(stim['auto_rewarded']).astype(bool)
    changeFlash = np.where(stim['is_change'] & ~autoRewarded)[0]
    notOmitted = [flash for flash in changeFlash if not any(stim[flash-1:flash+10]['omitted']) and flash+10<spikes.shape[1]]
    
    for region in regions:
        inRegion = np.array(units['structure_acronym']==region)
        if not any(inRegion):
            continue
        for layer in layers:
            print('session '+str(sessionIndex+1)+', '+region+', '+str(layer))
            if 'VIS' in region:
                inLayer = inRegion & np.in1d(units['cortical_layer'],layer)
            elif '1' not in layer:
                break
            else:
                inLayer = inRegion
            if not any(inLayer):
                continue
            s = np.zeros((inLayer.sum(),spikes.shape[1],spikes.shape[2]),dtype=bool)
            for i,u in enumerate(np.where(inLayer)[0]):
                s[i]=spikes[u,:,:]
                
            changeSp = s[:,changeFlash,:]
            preChangeSp = s[:,changeFlash-1,:]
            hasResp = findResponsiveUnits(preChangeSp,changeSp,baseWin,respWin)
            if not any(hasResp):
                continue
            base = s[hasResp,:,baseWin].sum(axis=2)
            resp = np.full((hasResp.sum(),len(stim)),np.nan)
            resp[:,1:] = s[hasResp,1:,respWin].sum(axis=2) - base[:,:-1]
            
            changeSpikes[region][layer].append(changeSp[hasResp].mean(axis=1))
            preChangeSpikes[region][layer].append(preChangeSp[hasResp].mean(axis=1))
            changeResp[region][layer].append(resp[:,changeFlash])
            preChangeResp[region][layer].append(resp[:,changeFlash-1])
            
            adaptSp = np.zeros((hasResp.sum(),len(notOmitted),11*750),dtype=bool)
            adaptR = np.zeros((hasResp.sum(),len(notOmitted),11))
            for i,flash in enumerate(notOmitted):
                adaptSp[:,i,:] = s[hasResp,flash-1:flash+10,:].reshape((hasResp.sum(),-1))
                adaptR[:,i,:] = resp[:,flash-1:flash+10]
            adaptSpikes[region][layer].append(adaptSp.mean(axis=1))
            adaptResp[region][layer].append(adaptR.mean(axis=1))
            
            flashCount = np.zeros(12)
            fb = np.zeros((hasResp.sum(),12))
            fr = fb.copy()
            changeCount = flashCount.copy()
            cb = fb.copy()
            cr = fb.copy()
            for i,row in stim.iterrows():
                lastLick = row['flashes_since_last_lick']
                if not np.isnan(lastLick) and lastLick<13:
                    ind = int(lastLick)-1
                    if not row['previous_omitted'] and lastLick < row['flashes_since_change']:
                        flashCount[ind] += 1
                        fb[:,ind] += base[:,i-1]
                        fr[:,ind] += resp[:,i]
                    if row['is_change'] and not row['auto_rewarded']:
                        changeCount[ind] += 1
                        cb[:,ind] += base[:,i-1]
                        cr[:,ind] += resp[:,i]
            fb /= flashCount
            fr /= flashCount
            cb /= changeCount
            cr /= changeCount
            flashBase[region][layer].append(fb)
            flashResp[region][layer].append(fr)
            changeFlashBase[region][layer].append(cb)
            changeFlashResp[region][layer].append(cr)
            

t = np.arange(11*750)/1000 - 0.75
flashTimes = np.arange(-0.75,7.5,0.75)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for region,clr in zip(regions,plt.cm.magma(np.linspace(0,0.8,len(regions)))):
    if 'VIS' in region:
        d = np.concatenate([np.concatenate(adaptSpikes[region][layer]) for layer in layers if len(adaptSpikes[region][layer])>0])
    else:
        d = np.concatenate(adaptSpikes[region][layers[0]])
    d -= d[:,baseWin].mean(axis=1)[:,None]
    d /= binSize
    ax.plot(t,d.mean(axis=0),color=clr,alpha=0.5,label=region+', n='+str(d.shape[0]))            
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlim([-0.25,7.5])
ax.set_xlabel('Time from change (s)')
ax.set_ylabel('Spikes/s')
ax.legend(loc='upper center')
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for region,clr in zip(regions,plt.cm.magma(np.linspace(0,0.8,len(regions)))):
    if 'VIS' in region:
        d = np.concatenate([np.concatenate(adaptResp[region][layer]) for layer in layers if len(adaptSpikes[region][layer])>0])
    else:
        d = np.concatenate(adaptResp[region][layers[0]])
    d /= d[:,1][:,None]
    d[np.isinf(d)] = np.nan
    mean = np.nanmean(d,axis=0)
    sem = np.nanstd(d,axis=0)/(d.shape[0]**0.5)
    ax.plot(flashTimes,mean,color=clr,alpha=0.5,label=region)
    for x,m,s in zip(flashTimes,mean,sem):
        ax.plot([x,x],[m-s,m+s],color=clr,alpha=0.5)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlim([-1,7.5])
ax.set_ylim([0,1.01])
ax.set_xlabel('Time from change (s)')
ax.set_ylabel('Norm. response')
ax.legend(loc='lower right')
plt.tight_layout()

fig = plt.figure(figsize=(6,11))
for i,region in enumerate(r for r in regions if 'VIS' in r):
    ax = fig.add_subplot(6,1,i+1)
    for layer,clr in zip(layers,plt.cm.magma(np.linspace(0,0.8,len(layers)))):
        if len(adaptSpikes[region][layer])>0:
            d = np.concatenate(adaptSpikes[region][layer])
            d -= d[:,baseWin].mean(axis=1)[:,None]
            d /= binSize
            ax.plot(t,d.mean(axis=0),color=clr,alpha=0.5,label=str(layer)+', n='+str(d.shape[0]))            
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xlim([-0.25,7.5])
    if i==5:
        ax.set_xlabel('Time from change (s)')
    ax.set_ylabel('Spikes/s')
    ax.legend(loc='upper center')
    ax.set_title(region)
plt.tight_layout()

fig = plt.figure(figsize=(6,11))
for i,region in enumerate(r for r in regions if 'VIS' in r):
    ax = fig.add_subplot(6,1,i+1)
    for layer,clr in zip(layers,plt.cm.magma(np.linspace(0,0.8,len(layers)))):
        if len(adaptSpikes[region][layer])>0:
            d = np.concatenate(adaptResp[region][layer])
            d /= d[:,1][:,None]
            d[np.isinf(d)] = np.nan
            mean = np.nanmean(d,axis=0)
            sem = np.nanstd(d,axis=0)/(d.shape[0]**0.5)
            ax.plot(flashTimes,mean,color=clr,alpha=0.5,label=layer)
            for x,m,s in zip(flashTimes,mean,sem):
                ax.plot([x,x],[m-s,m+s],color=clr,alpha=0.5)            
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xlim([-1,7.5])
    ax.set_ylim([0,1.01])
    if i==5:
        ax.set_xlabel('Time from change (s)')
    if i==0:
        ax.set_ylabel('Norm. response')
        ax.legend(loc='upper center')
    ax.set_title(region)
plt.tight_layout()

fig = plt.figure(figsize=(6,8))
xticks = np.arange(len(regions))
for i,layer in enumerate(layers):
    ax = fig.add_subplot(4,1,i+1)
    mean = []
    sem = []
    for region in regions:
        if 'VIS' in region:
            d = np.concatenate(adaptResp[region][layer])
        else:
            d = np.concatenate(adaptResp[region][layers[0]])
        d /= d[:,1][:,None]
        d[np.isinf(d)] = np.nan
        mean.append(np.nanmean(d[:,-1],axis=0))
        sem.append(np.nanstd(d[:,-1],axis=0)/(d.shape[0]**0.5))
    ax.plot(xticks,mean,color='k')
    for x,m,s in zip(xticks,mean,sem):
        ax.plot([x,x],[m-s,m+s],color='k')
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xticks(xticks)
    ax.set_xticklabels(regions)
    ax.set_ylim([0,1.01])
    ax.set_ylabel('Adaptation ratio')
    ax.set_title('cortical layer '+str(layer))
plt.tight_layout()


flashSinceLickTimes = np.arange(0.75,0.75*13,0.75)
for r,ylbl in zip((flashResp,flashBase,changeFlashResp,changeFlashBase),('flash resp','pre-flash baseline','change resp','pre-change baseline')):
    fig = plt.figure(figsize=(6,8))
    for i,layer in enumerate(layers):
        ax = fig.add_subplot(4,1,i+1)
        ymax = 0
        for region,clr in zip(regions,plt.cm.magma(np.linspace(0,0.8,len(regions)))):
            if 'VIS' in region:
                d = np.concatenate(r[region][layer])
            else:
                d = np.concatenate(r[region][layers[0]])
            d /= (respWin.stop-respWin.start)/1000
            mean = np.nanmean(d,axis=0)
            sem = np.nanstd(d,axis=0)/(d.shape[0]**0.5)
            lbl = region if i==0 else None
            ax.plot(flashSinceLickTimes,mean,color=clr,alpha=0.5,label=lbl)
            for x,m,s in zip(flashSinceLickTimes,mean,sem):
                ax.plot([x,x],[m-s,m+s],color=clr,alpha=0.5)
            ymax = max(ymax,np.nanmax(mean+sem))
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_xlim([0,9.75])
        ax.set_ylim([0,ymax])
        ax.set_xlabel('time since lick (s)')
        ax.set_ylabel(ylbl+'\n(spikes/s)')
        if i==0:
            loc = 'upper left' if 'change' in ylbl else 'upper right'
            ax.legend(loc=loc)
        ax.set_title('cortical layer '+str(layer))
    plt.tight_layout()


## decoding
def crossValidate(model,X,y,nSplits):
    # cross validation using stratified shuffle split
    # each split preserves the percentage of samples of each class
    # all samples used in one test set
    
    classVals = np.unique(y)
    nClasses = len(classVals)
    nSamples = len(y)
    samplesPerClass = [np.sum(y==val) for val in classVals]
    samplesPerSplit = [round(n/nSplits) for n in samplesPerClass]
    shuffleInd = np.random.permutation(nSamples)
    
    cv = {'estimator': [sklearn.base.clone(model) for _ in range(nSplits)]}
    cv['train_score'] = []
    cv['test_score'] = []
    cv['predict'] = np.full(nSamples,np.nan)
    cv['predict_proba'] = np.full((nSamples,nClasses),np.nan)
    cv['decision_function'] = np.full((nSamples,nClasses),np.nan) if nClasses>2 else np.full(nSamples,np.nan)
    cv['feature_importance'] = []
    cv['coef'] = []
    modelMethods = dir(model)
    
    for k,estimator in enumerate(cv['estimator']):
        testInd = []
        for val,n in zip(classVals,samplesPerSplit):
            start = k*n
            ind = shuffleInd[y[shuffleInd]==val]
            testInd.extend(ind[start:start+n] if k+1<nSplits else ind[start:])
        trainInd = np.setdiff1d(shuffleInd,testInd)
        estimator.fit(X[trainInd],y[trainInd])
        cv['train_score'].append(estimator.score(X[trainInd],y[trainInd]))
        cv['test_score'].append(estimator.score(X[testInd],y[testInd]))
        cv['predict'][testInd] = estimator.predict(X[testInd])
        for method in ('predict_proba','decision_function'):
            if method in modelMethods:
                cv[method][testInd] = getattr(estimator,method)(X[testInd])
        for attr in ('feature_importance_','coef_'):
            if attr in estimator.__dict__:
                cv[attr[:-1]].append(getattr(estimator,attr))
    return cv


sessionIds = stimTable['session_id'].unique()

regions = ('LGd','VISp','VISl','VISrl','VISal','VISpm','VISam','LP',
           'MRN','MB',('SCig','SCiw'),'APN','NOT',
           ('HPF','DG','CA1','CA3'),('SUB','ProS','PRE','POST'))
layers = ('all',('1','2/3'),'4','5',('6a','6b'))

model = sklearn.svm.LinearSVC(C=1.0,max_iter=1e4)

unitSampleSize = np.arange(5,45,5)

nCrossVal = 5

decodeWindowSize = 10
decodeWindowEnd = 500 # respWin.stop
decodeWindows = np.arange(decodeWindowSize,decodeWindowEnd+decodeWindowSize,decodeWindowSize)

decodeData = {sessionId: {region: {layer: {sampleSize: {} for sampleSize in unitSampleSize} for layer in layers} for region in regions} for sessionId in sessionIds}
for sessionId in sessionIds:
    for region in regions:
        for layer in layers:
            decodeData[sessionId][region][layer]['nUnits'] = 0

warnings.filterwarnings('ignore')
for sessionIndex,sessionId in enumerate(sessionIds):
    units = unitTable.set_index('unit_id').loc[unitData[str(sessionId)]['unitIds'][:]]
    spikes = unitData[str(sessionId)]['spikes']
    
    stim = stimTable[(stimTable['session_id']==sessionId) & stimTable['active']].reset_index()
    autoRewarded = np.array(stim['auto_rewarded']).astype(bool)
    changeFlash = np.where(stim['is_change'] & ~autoRewarded)[0]
    isCatch = stim['catch']
    isCatch[isCatch.isnull()] = False
    isCatch = np.array(isCatch).astype(bool)
    catchFlash = np.searchsorted(stim['start_time'],np.unique(stim['change_time_no_display_delay'][isCatch]))
    changeTimes = np.array(stim['start_time'][changeFlash])
    catchTimes = np.array(stim['start_time'][catchFlash])
    hit = np.array(stim['hit'][changeFlash])
    falseAlarm = np.array(stim['false_alarm'][catchFlash])
    
    engagedChange,engagedCatch = [np.array([np.sum(hit[(changeTimes>t-60) & (changeTimes<t+60)]) > 1 for t in times]) for times in (changeTimes,catchTimes)]
    nChangeTrials = engagedChange.sum()
    nCatchTrials = engagedCatch.sum()
    
    decodeData[sessionId]['changeBehav'] = hit[engagedChange]
    decodeData[sessionId]['catchBehav'] = falseAlarm[engagedCatch]
    
    for region in regions:
        inRegion = np.in1d(units['structure_acronym'],region)
        if not any(inRegion):
            continue
        for layer in layers:
            print('session '+str(sessionIndex+1)+', '+str(region)+', '+str(layer))
            if layer=='all':
                inLayer = inRegion
            elif 'VIS' in region:
                inLayer = inRegion & np.in1d(units['cortical_layer'],layer)
            else:
                continue
            if not any(inLayer):
                continue
            sp = np.zeros((inLayer.sum(),spikes.shape[1],spikes.shape[2]),dtype=bool)
            for i,u in enumerate(np.where(inLayer)[0]):
                sp[i]=spikes[u,:,:]
                
            changeSp = sp[:,changeFlash[engagedChange],:]
            preChangeSp = sp[:,changeFlash[engagedChange]-1,:]
            hasResp = findResponsiveUnits(preChangeSp,changeSp,baseWin,respWin)
            nUnits = hasResp.sum()
            decodeData[sessionId][region][layer]['nUnits'] = nUnits
            if not any(hasResp):
                continue
            
            base = sp[hasResp,:,baseWin].sum(axis=2)
            resp = np.full((hasResp.sum(),len(stim)),np.nan)
            resp[:,1:] = sp[hasResp,1:,respWin].sum(axis=2) - base[:,:-1]
            decodeData[sessionId][region][layer]['changeResp'] = resp[:,changeFlash[engagedChange]]
            decodeData[sessionId][region][layer]['preChangeResp'] = resp[:,changeFlash[engagedChange]-1]
            
            changeSp,preChangeSp = [s[hasResp,:,:decodeWindows[-1]].reshape((nUnits,nChangeTrials,len(decodeWindows),decodeWindowSize)).sum(axis=-1) for s in (changeSp,preChangeSp)]
            catchSp = sp[hasResp][:,catchFlash[engagedCatch],:decodeWindows[-1]].reshape((nUnits,nCatchTrials,len(decodeWindows),decodeWindowSize)).sum(axis=-1)
            
            for sampleSize in unitSampleSize:
                if nUnits < sampleSize:
                    continue
                if sampleSize>1:
                    if sampleSize==nUnits:
                        nSamples = 1
                        unitSamples = [np.arange(nUnits)]
                    else:
                        # >99% chance each neuron is chosen at least once
                        nSamples = int(math.ceil(math.log(0.01)/math.log(1-sampleSize/nUnits)))
                        unitSamples = [np.random.choice(nUnits,sampleSize,replace=False) for _ in range(nSamples)]
                else:
                    nSamples = nUnits
                    unitSamples = [[i] for i in range(nUnits)]
                changeTrainAccuracy = np.full((len(unitSamples),len(decodeWindows)),np.nan)
                changeFeatureWeights = np.full((len(unitSamples),len(decodeWindows),nUnits,len(decodeWindows)),np.nan)
                changeAccuracy = changeTrainAccuracy.copy()
                changePrediction = np.full((len(unitSamples),len(decodeWindows),nChangeTrials),np.nan)
                changeConfidence = changePrediction.copy()
                catchAccuracy = changeAccuracy.copy()
                catchPrediction = np.full((len(unitSamples),len(decodeWindows),nCatchTrials),np.nan)
                catchConfidence = catchPrediction.copy()
                lickTrainAccuracy = changeAccuracy.copy()
                lickFeatureWeights = changeFeatureWeights.copy()
                lickAccuracy = changeAccuracy.copy()
                lickBalancedAccuracy = changeAccuracy.copy()
                lickPrediction = np.full((len(unitSamples),len(decodeWindows),nChangeTrials+nCatchTrials),np.nan)
                lickConfidence = lickPrediction.copy()
                for i,unitSamp in enumerate(unitSamples):
                    for j,winEnd in enumerate((decodeWindows/decodeWindowSize).astype(int)):
                        if sampleSize!=20 and winEnd*decodeWindowSize!=respWin.stop:
                            continue
                        Xchange = np.concatenate([s[unitSamp,:,:winEnd].transpose(1,0,2).reshape((nChangeTrials,-1)) for s in (changeSp,preChangeSp)])
                        Ychange = np.zeros(Xchange.shape[0])
                        Ychange[:nChangeTrials] = 1                        
                        cv = crossValidate(model,Xchange,Ychange,nCrossVal)
                        changeTrainAccuracy[i,j] = np.mean(cv['train_score'])
                        changeFeatureWeights[i,j,unitSamp,:winEnd] = np.mean(cv['coef'],axis=0).reshape(sampleSize,winEnd)
                        changeAccuracy[i,j] = np.mean(cv['test_score'])
                        changePrediction[i,j] = cv['predict'][:nChangeTrials]
                        changeConfidence[i,j] = cv['decision_function'][:nChangeTrials]
                        Xcatch = catchSp[unitSamp,:,:winEnd].transpose(1,0,2).reshape((nCatchTrials,-1))
                        catchAccuracy[i,j] = np.mean([estimator.score(Xcatch,np.zeros(nCatchTrials)) for estimator in cv['estimator']])
                        catchPrediction[i,j] = scipy.stats.mode([estimator.predict(Xcatch) for estimator in cv['estimator']],axis=0)[0][0]
                        catchConfidence[i,j] = np.mean([estimator.decision_function(Xcatch) for estimator in cv['estimator']],axis=0)
                        Xlick = np.concatenate((Xchange[:nChangeTrials],Xcatch))
                        Ylick = np.concatenate((hit[engagedChange],falseAlarm[engagedCatch]))
                        cv = crossValidate(model,Xlick,Ylick,nCrossVal)
                        lickTrainAccuracy[i,j] = np.mean(cv['train_score'])
                        lickFeatureWeights[i,j,unitSamp,:winEnd] = np.mean(cv['coef'],axis=0).reshape(sampleSize,winEnd)
                        lickAccuracy[i,j] = np.mean(cv['test_score'])
                        lickBalancedAccuracy[i,j] = sklearn.metrics.balanced_accuracy_score(Ylick,cv['predict'])
                        lickPrediction[i,j] = cv['predict']
                        lickConfidence[i,j] = cv['decision_function']
                decodeData[sessionId][region][layer][sampleSize]['changeTrainAccuracy'] = np.median(changeTrainAccuracy,axis=0)
                decodeData[sessionId][region][layer][sampleSize]['changeFeatureWeights'] = np.nanmedian(changeFeatureWeights,axis=0)
                decodeData[sessionId][region][layer][sampleSize]['changeAccuracy'] = np.median(changeAccuracy,axis=0) 
                decodeData[sessionId][region][layer][sampleSize]['changePrediction'] = scipy.stats.mode(changePrediction,axis=0)[0][0]
                decodeData[sessionId][region][layer][sampleSize]['changeConfidence'] = np.median(changeConfidence,axis=0)
                decodeData[sessionId][region][layer][sampleSize]['catchAccuracy'] = np.median(catchAccuracy,axis=0) 
                decodeData[sessionId][region][layer][sampleSize]['catchPrediction'] = scipy.stats.mode(catchPrediction,axis=0)[0][0] 
                decodeData[sessionId][region][layer][sampleSize]['catchConfidence'] = np.median(catchConfidence,axis=0)
                decodeData[sessionId][region][layer][sampleSize]['lickTrainAccuracy'] = np.median(lickTrainAccuracy,axis=0)
                decodeData[sessionId][region][layer][sampleSize]['lickFeatureWeights'] = np.nanmedian(lickFeatureWeights,axis=0)
                decodeData[sessionId][region][layer][sampleSize]['lickAccuracy'] = np.median(lickAccuracy,axis=0)
                decodeData[sessionId][region][layer][sampleSize]['lickBalancedAccuracy'] = np.median(lickBalancedAccuracy,axis=0) 
                decodeData[sessionId][region][layer][sampleSize]['lickPrediction'] = scipy.stats.mode(lickPrediction,axis=0)[0][0]
                decodeData[sessionId][region][layer][sampleSize]['lickConfidence'] = np.median(lickConfidence,axis=0)
warnings.filterwarnings('default')

# save result to pkl file
pkl = fileIO.saveFile(fileType='*.pkl')
pickle.dump(decodeData,open(pkl,'wb'))

# get result from pkl file
pkl = fileIO.getFile(fileType='*.pkl')
decodeData = pickle.load(open(pkl,'rb'))


for region in regions:
    for layer in layers:
        print(region,layer)
        print([decodeData[sessionId][region][layer]['nUnits'] for sessionId in sessionIds])
        print('\n')
        
winInd = np.where(decodeWindows==respWin.stop)[0][0]
        
fig = plt.figure(figsize=(8,8))       
for i,region in enumerate(regions):
    ax = fig.add_subplot(3,3,i+1)
    y = 1
    for layer,clr in zip(layers,plt.cm.magma(np.linspace(0,0.8,len(layers)))):
        if layer==layers[0] or 'VIS' in region:
            mean = []
            sem = []
            nUnits = []
            for sampleSize in unitSampleSize:
                lyr = layer if 'VIS' in region else layers[0]
                d = [decodeData[sessionId][region][lyr][sampleSize]['changeCatchAccuracy'][winInd] for sessionId in sessionIds if len(decodeData[sessionId][region][lyr][sampleSize])>0]
                mean.append(np.mean(d))
                sem.append(np.std(d)/(len(d)**0.5))
                nUnits.append(len(d))
            ax.plot(unitSampleSize,mean,'-o',color=clr,mfc='none')
            for x,m,s,n in zip(unitSampleSize,mean,sem,nUnits):
                if n>0:
                    ax.plot([x,x],[m-s,m+s],color=clr)
                    ax.text(x,y,str(n),color=clr,fontsize=6,ha='center',va='top')
        y -= 0.045
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xlim([0,1.05*max(unitSampleSize)])
    ax.set_ylim([0.5,1])
    if i==3:
        ax.set_ylabel('Change decoding accuracy')
    if i==7:
        ax.set_xlabel('Number of neurons')
    ax.set_title(region)
plt.tight_layout()

sampleSize = 20

fig = plt.figure(figsize=(10,8))
for i,layer in enumerate(layers):
    ax = fig.add_subplot(2,2,i+1)
    for region,clr in zip(regions,plt.cm.magma(np.linspace(0,0.8,len(regions)))):
        lyr = layer if 'VIS' in region else layers[0]
        d = [decodeData[sessionId][region][lyr][sampleSize]['changeAccuracy'] for sessionId in sessionIds if len(decodeData[sessionId][region][lyr][sampleSize])>0]
        if len(d)>0:
            m = np.mean(d,axis=0)
            s = np.std(d,axis=0)/(len(d)**0.5)
            ax.plot(decodeWindows-decodeWindowSize/2,m,color=clr,label=region)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_ylim([0.4,1])
    ax.set_xlabel('Time from change (ms)')
    ax.set_ylabel('Change decoding accuracy')
    if i==0:
        ax.legend(loc='upper left')
    ax.set_title('cortical layer '+str(layer))
plt.tight_layout()

fig = plt.figure(figsize=(10,8))
xticks = np.arange(len(regions))
for i,layer in enumerate(layers):
    ax = fig.add_subplot(2,2,i+1)
    for x,region in enumerate(regions):
        lyr = layer if 'VIS' in region else layers[0]
        d = [decodeData[sessionId][region][lyr][sampleSize]['changeAccuracy'][winInd] for sessionId in sessionIds if len(decodeData[sessionId][region][lyr][sampleSize])>0]
        m = np.mean(d)
        s = np.std(d)/(len(d)**0.5)
        ax.plot(x,m,'ko')
        ax.plot([x,x],[m-s,m+s],'k')
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xticks(xticks)
    ax.set_xticklabels(regions)
    ax.set_ylim([0.5,1])
    ax.set_ylabel('Change decoding accuracy')
    ax.set_title('cortical layer '+str(layer))
plt.tight_layout()
    
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for region,clr in zip(regions,plt.cm.magma(np.linspace(0,0.8,len(regions)))):
    d = [decodeData[sessionId][region][layer][sampleSize]['changeFeatureWeights'][-1] for sessionId in sessionIds for layer in layers if len(decodeData[sessionId][region][layer][sampleSize])>0]
    if len(d)>0:
        d = np.concatenate(d)
        m = np.nanmean(d,axis=0)
        s = np.std(d,axis=0)/(len(d)**0.5)
        ax.plot(decodeWindows-decodeWindowSize/2,m,color=clr,label=region)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlabel('Time from change (ms)')
ax.set_ylabel('Decoder weighting')
ax.legend()
plt.tight_layout()

fig = plt.figure(figsize=(10,8))
for i,layer in enumerate(layers):
    ax = fig.add_subplot(2,2,i+1)
    for region,clr in zip(regions,plt.cm.magma(np.linspace(0,0.8,len(regions)))):
        lyr = layer if 'VIS' in region else layers[0]
        w = []
        cmi = []
        for sessionId in sessionIds:
            if len(decodeData[sessionId][region][lyr][sampleSize])>0:
                w.append(np.nanmax(decodeData[sessionId][region][lyr][sampleSize]['changeFeatureWeights'][winInd],axis=1))
                changeResp,preChangeResp = [decodeData[sessionId][region][lyr][resp].mean(axis=1) for resp in ('changeResp','preChangeResp')]
                cmi.append((changeResp - preChangeResp) / (changeResp + preChangeResp))
        if len(w)>0:
            ax.plot(np.concatenate(cmi),np.concatenate(w),'o',mec=clr,mfc='none',alpha=0.25,label=region)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xlabel('Change modulation index')
    ax.set_ylabel('Decoder weighting')
    ax.legend(loc='upper left')
    ax.set_title('cortical layer '+str(layer))
plt.tight_layout()

fig = plt.figure(figsize=(10,8))
for i,layer in enumerate(layers):
    ax = fig.add_subplot(2,2,i+1)
    for region,clr in zip(regions,plt.cm.magma(np.linspace(0,0.8,len(regions)))): 
        lyr = layer if 'VIS' in region else layers[0]
        r = []
        for sessionId in sessionIds:
            if len(decodeData[sessionId][region][lyr][sampleSize])>0:
                b = np.concatenate([decodeData[sessionId][resp] for resp in ('changeBehav','catchBehav')])
                r.append([])
                for j,_ in enumerate(decodeWindows):
                    d = np.concatenate([decodeData[sessionId][region][lyr][sampleSize][conf][j] for conf in ('changeConfidence','catchConfidence')])
                    r[-1].append(np.corrcoef(b,d)[0,1])
        if len(r)>0:
            m = np.mean(r,axis=0)
            s = np.std(r,axis=0)/(len(r)**0.5)
            ax.plot(decodeWindows-decodeWindowSize/2,m,color=clr,label=region)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_ylim([-0.1,0.8])
    ax.set_xlabel('Time from change (ms)')
    ax.set_ylabel('Correlation of decoder confidence and behavior')
    if i==0:
        ax.legend(loc='upper right')
    ax.set_title('cortical layer '+str(layer))
plt.tight_layout()

fig = plt.figure(figsize=(10,8))
for i,layer in enumerate(layers):
    ax = fig.add_subplot(2,2,i+1)
    for x,region in enumerate(regions):
        lyr = layer if 'VIS' in region else layers[0]
        r = []
        for sessionId in sessionIds:
            if len(decodeData[sessionId][region][lyr][sampleSize])>0:
                b = np.concatenate([decodeData[sessionId][resp] for resp in ('changeBehav','catchBehav')])
                d = np.concatenate([decodeData[sessionId][region][lyr][sampleSize][conf][winInd] for conf in ('changeConfidence','catchConfidence')])
                r.append(np.corrcoef(b,d)[0,1])
        m = np.mean(r)
        s = np.std(r)/(len(r)**0.5)
        ax.plot(x,m,'ko')
        ax.plot([x,x],[m-s,m+s],'k')
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xticks(xticks)
    ax.set_xticklabels(regions)
    ax.set_ylim([0,0.5])
    ax.set_ylabel('Correlation of decoder confidence and behavior')
    ax.set_title('cortical layer '+str(layer))
plt.tight_layout()
    
fig = plt.figure(figsize=(10,8))
for i,layer in enumerate(layers):
    ax = fig.add_subplot(2,2,i+1)
    for x,region in enumerate(regions):
        lyr = layer if 'VIS' in region else layers[0]
        js = []
        for sessionId in sessionIds:
            if len(decodeData[sessionId][region][lyr][sampleSize])>0:
                b = np.concatenate([decodeData[sessionId][resp] for resp in ('changeBehav','catchBehav')])
                d = np.concatenate([decodeData[sessionId][region][lyr][sampleSize][pred][winInd] for pred in ('changePrediction','catchPrediction')]).astype(bool)
                js.append((b & d).sum() / (b | d).sum())
        m = np.mean(js)
        s = np.std(js)/(len(js)**0.5)
        ax.plot(x,m,'ko')
        ax.plot([x,x],[m-s,m+s],'k')
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xticks(xticks)
    ax.set_xticklabels(regions)
    ax.set_ylim([0.4,0.8])
    ax.set_ylabel('Jaccard similarity between decoder and behavior')
    ax.set_title('cortical layer '+str(layer))
plt.tight_layout()
  
fig = plt.figure(figsize=(10,8))
alim = [-0.5,4]  
for i,layer in enumerate(layers):
    ax = fig.add_subplot(2,2,i+1)
    ax.plot(alim,alim,'--',color='0.75')
    for region,clr in zip(regions,plt.cm.magma(np.linspace(0,0.8,len(regions)))):
        lyr = layer if 'VIS' in region else layers[0]
        d = []
        for sessionId in sessionIds:
            if len(decodeData[sessionId][region][lyr][sampleSize])>0:
                changeBehav = decodeData[sessionId]['changeBehav']
                catchBehav = decodeData[sessionId]['catchBehav']
                changePred = decodeData[sessionId][region][lyr][sampleSize]['changePrediction'][winInd]
                catchPred = decodeData[sessionId][region][lyr][sampleSize]['catchPrediction'][winInd]
                d.append([calcDprime(change.sum()/change.size,catch.sum()/catch.size,change.size,catch.size) for change,catch in zip((changeBehav,changePred),(catchBehav,catchPred))])        
        if len(d)>0:
            d = np.array(d)
            ax.plot(d[:,0],d[:,1],'o',mec=clr,mfc='none',label=region)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xticks(np.arange(5))
    ax.set_yticks(np.arange(5))
    ax.set_xlim(alim)
    ax.set_ylim(alim)
    ax.set_aspect('equal')
    ax.set_xlabel('d\' behavior')
    ax.set_ylabel('d\' decoder')
    if i==0:
        ax.legend(loc='upper left')
    ax.set_title('cortical layer '+str(layer))
plt.tight_layout()

# lick decoding plots



