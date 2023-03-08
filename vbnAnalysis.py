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
import scipy.cluster
import pandas as pd
import h5py
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
import sklearn
from sklearn.svm import LinearSVC
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

# h5Path = r'C:/Users/svc_ccg/Desktop/Analysis/vbn/vbnAllUnitSpikeTensor.hdf5'
h5Path = r'C:/Users/svc_ccg/Desktop/Analysis/vbn/vbnAllUnitSpikeTensor_passive.hdf5'
h5File = h5py.File(h5Path,'w')

sessionCount = 0
for sessionId,sessionData in sessions.iterrows():
    sessionCount += 1
    print('session '+str(sessionCount))
    
    session = cache.get_ecephys_session(ecephys_session_id=sessionId)
    
    stim = session.stimulus_presentations
    # flashTimes = stim.start_time[stim.active]
    flashTimes = stim.start_time[stim.stimulus_block==5] # passive
    
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

unitDataPassive = h5py.File(os.path.join(baseDir,'vbnAllUnitSpikeTensor_passive.hdf5'),mode='r')

sessionIds = stimTable['session_id'].unique()
# sessionIds = stimTable['session_id'][stimTable['experience_level']=='Familiar'].unique()

binSize = 0.001
baseWin = slice(680,750)
respWin = slice(30,100)

regions = ('LGd','VISp','VISl','VISrl','VISal','VISpm','VISam','LP',
           'MRN','MB',('SCig','SCiw'),'APN','NOT',
           ('HPF','DG','CA1','CA3'),('SUB','ProS','PRE','POST'))
layers = ('all',('1','2/3'),'4','5',('6a','6b'))
regionColors = plt.cm.tab20(np.linspace(0,1,len(regions)))
layerColors = plt.cm.magma(np.linspace(0,0.8,len(layers)))
regionLabels = []
for region in regions:
    if 'SCig' in region:
        regionLabels.append('SC')
    elif 'HPF' in region:
        regionLabels.append('Hipp')
    elif 'SUB' in region:
        regionLabels.append('Sub')
    else:
        regionLabels.append(region)


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

allRegions = np.unique(unitTable['structure_acronym'])

nUnits = {region: [] for region in allRegions}

for sessionIndex,sessionId in enumerate(sessionIds):
    print('session '+str(sessionIndex+1))
    units = unitTable.set_index('unit_id').loc[unitData[str(sessionId)]['unitIds'][:]]
    spikes = unitData[str(sessionId)]['spikes']
    
    stim = stimTable[(stimTable['session_id']==sessionId) & stimTable['active']].reset_index()
    autoRewarded = np.array(stim['auto_rewarded']).astype(bool)
    changeFlash = np.where(stim['is_change'] & ~autoRewarded)[0]
    
    for region in allRegions:
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
    print(sum([n>29 for n in nUnits[region]]))

# save result to pkl file
pkl = fileIO.saveFile(fileType='*.pkl')
pickle.dump(nUnits,open(pkl,'wb'))

# get result from pkl file
pkl = fileIO.getFile(fileType='*.pkl')
nUnits = pickle.load(open(pkl,'rb'))


## psth
respLabels = ('hit','miss','false alarm','correct reject')
psthBinSize = 5
psthEnd = 500
psthBins = np.arange(psthBinSize,psthEnd+psthBinSize,psthBinSize)
psth = {sessionId: {region: {layer: {state: {resp: [] for resp in respLabels} for state in ('active','passive')} for layer in layers} for region in regions} for sessionId in sessionIds}
base = copy.deepcopy(psth)
for sessionIndex,sessionId in enumerate(sessionIds):
    units = unitTable.set_index('unit_id').loc[unitData[str(sessionId)]['unitIds'][:]]
    spikes = unitData[str(sessionId)]['spikes']
    spikesPassive = unitDataPassive[str(sessionId)]['spikes']
    
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
    
    engagedChange,engagedCatch = [np.array([np.sum(hit[(changeTimes>t-60) & (changeTimes<t+60)]) > 1 for t in times]) for times in (changeTimes,catchTimes)]
    changeFlash = changeFlash[engagedChange]
    catchFlash = catchFlash[engagedCatch]
    nChangeTrials = engagedChange.sum()
    nCatchTrials = engagedCatch.sum()
    
    hit = hit[engagedChange]
    falseAlarm = np.array(stim['false_alarm'][catchFlash])
    
    for region in regions:
        if not (region=='MRN' or 'SCig' in region):
            continue
        inRegion = np.in1d(units['structure_acronym'],region)
        if not any(inRegion):
            continue
        for layer in ('all',): # layers
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
                
            changeSp = sp[:,changeFlash,:]
            preChangeSp = sp[:,changeFlash-1,:]
            hasResp = findResponsiveUnits(preChangeSp,changeSp,baseWin,respWin)
            nUnits = hasResp.sum()
            if not any(hasResp):
                continue
            
            for state in ('active','passive'):
                if state=='passive':
                    for i,u in enumerate(np.where(inLayer)[0]):
                        sp[i]=spikesPassive[u,:,:]
                
                changeSp = sp[hasResp][:,changeFlash,:psthEnd].reshape((nUnits,nChangeTrials,len(psthBins),psthBinSize)).sum(axis=-1)
                catchSp = sp[hasResp][:,catchFlash,:psthEnd].reshape((nUnits,nCatchTrials,len(psthBins),psthBinSize)).sum(axis=-1)
                b = sp[hasResp,:,baseWin].sum(axis=2)
                
                for s,flash,lick,lbls in zip((changeSp,catchSp),(changeFlash,catchFlash),(hit,falseAlarm),(('hit','miss'),('false alarm','correct reject'))):
                    for r,lbl in zip((lick,~lick),lbls): 
                        psth[sessionId][region][layer][state][lbl].append(np.mean(s[:,r],axis=1)/psthBinSize*1000)
                        base[sessionId][region][layer][state][lbl].append(np.mean(b[:,flash[r]-1],axis=1)/(baseWin.stop-baseWin.start)/binSize)


baseSubtract = True
layer = 'all' 
x = psthBins-psthBinSize/2
for state in ('active','passive'):
    fig = plt.figure(figsize=(16,8))    
    for i,(region,lbl) in enumerate(zip(regions,regionLabels)):
        ax = fig.add_subplot(3,5,i+1)
        for resp,clr in zip(respLabels,'krgb'):
            d = np.concatenate([psth[sessionId][region][layer][state][resp][0] for sessionId in sessionIds if len(psth[sessionId][region][layer][state][resp])>0])
            if baseSubtract:
                b = np.concatenate([base[sessionId][region][layer][state][resp][0] for sessionId in sessionIds if len(base[sessionId][region][layer][state][resp])>0])
                d -= b[:,None]
            m = np.nanmean(d,axis=0)
            s = np.nanstd(d,axis=0)/(len(d)**0.5)
            ax.plot(x,m,color=clr,label=resp)
            ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_xlabel('Time from change/catch (ms)')
        ax.set_ylabel('Spikes/s')
        if i==0:
            ax.legend()
        ax.set_title(lbl+' (n='+str(len(d))+'), '+state)
    plt.tight_layout()

for resp in respLabels:    
    fig = plt.figure(figsize=(16,8))   
    for i,(region,lbl) in enumerate(zip(regions,regionLabels)):
        ax = fig.add_subplot(3,5,i+1)
        for state,clr in zip(('active','passive'),'gm'):
            d = np.concatenate([psth[sessionId][region][layer][state][resp][0] for sessionId in sessionIds if len(psth[sessionId][region][layer][state][resp])>0])
            if baseSubtract:
                b = np.concatenate([base[sessionId][region][layer][state][resp][0] for sessionId in sessionIds if len(psth[sessionId][region][layer][state][resp])>0])
                d -= b[:,None]
            m = np.nanmean(d,axis=0)
            s = np.nanstd(d,axis=0)/(len(d)**0.5)
            ax.plot(x,m,color=clr,label=state)
            ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_xlabel('Time from change/catch (ms)')
        ax.set_ylabel('Spikes/s')
        if i==0:
            ax.legend()
        ax.set_title(lbl+' (n='+str(len(d))+'), '+resp)
    plt.tight_layout()
    
    
# cluster units by psth

def cluster(data,nClusters=None,method='ward',metric='euclidean',plot=False,colors=None,nreps=1000,labels=None):
    # data is n samples x m parameters
    linkageMat = scipy.cluster.hierarchy.linkage(data,method=method,metric=metric)
    if nClusters is None:
        clustId = None
    else:
        clustId = scipy.cluster.hierarchy.fcluster(linkageMat,nClusters,'maxclust')
    if plot:
        plt.figure(facecolor='w')
        ax = plt.subplot(1,1,1)
        colorThresh = 0 if nClusters<2 else linkageMat[::-1,2][nClusters-2]
        if colors is not None:
            scipy.cluster.hierarchy.set_link_color_palette(list(colors))
        if labels=='off':
            labels=None
            noLabels=True
        else:
            noLabels=False
        scipy.cluster.hierarchy.dendrogram(linkageMat,ax=ax,color_threshold=colorThresh,above_threshold_color='k',labels=labels,no_labels=noLabels)
        scipy.cluster.hierarchy.set_link_color_palette(None)
        ax.set_yticks([])
        for side in ('right','top','left','bottom'):
            ax.spines[side].set_visible(False)
        plt.tight_layout()
        
        if nreps>0:
            randLinkage = np.zeros((nreps,linkageMat.shape[0]))
            shuffledData = data.copy()
            for i in range(nreps):
                for j in range(data.shape[1]):
                    shuffledData[:,j] = data[np.random.permutation(data.shape[0]),j]
                _,m = cluster(shuffledData,method=method,metric=metric)
                randLinkage[i] = m[::-1,2]
            
            plt.figure(facecolor='w')
            ax = plt.subplot(1,1,1)
            k = np.arange(linkageMat.shape[0])+2
            ax.plot(k,np.percentile(randLinkage,2.5,axis=0),'k--')
            ax.plot(k,np.percentile(randLinkage,97.5,axis=0),'k--')
            ax.plot(k,linkageMat[::-1,2],'ko-',mfc='none',ms=10,mew=2,linewidth=2)
            ax.set_xlim([0,k[-1]+1])
            ax.set_xlabel('Cluster')
            ax.set_ylabel('Linkage Distance')
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False)
            plt.tight_layout()
    
    return clustId,linkageMat

def pca(data,plot=False):
    # data is n samples x m parameters
    eigVal,eigVec = np.linalg.eigh(np.cov(data,rowvar=False))
    order = np.argsort(eigVal)[::-1]
    eigVal = eigVal[order]
    eigVec = eigVec[:,order]
    pcaData = data.dot(eigVec)
    if plot:
        fig = plt.figure(facecolor='w')
        ax = fig.add_subplot(1,1,1)
        ax.plot(np.arange(1,eigVal.size+1),eigVal.cumsum()/eigVal.sum(),'k')
        ax.set_xlim((0.5,eigVal.size+0.5))
        ax.set_ylim((0,1.02))
        ax.set_xlabel('PC')
        ax.set_ylabel('Cumulative Fraction of Variance')
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        
        fig = plt.figure(facecolor='w')
        ax = fig.add_subplot(1,1,1)
        im = ax.imshow(eigVec,clim=(-1,1),cmap='bwr',interpolation='none',origin='lower')
        ax.set_xlabel('PC')
        ax.set_ylabel('Parameter')
        ax.set_title('PC Weightings')
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        cb = plt.colorbar(im,ax=ax,fraction=0.05,pad=0.04,shrink=0.5)
        cb.ax.tick_params(length=0)
        cb.set_ticks([-1,0,1])
    return pcaData,eigVal,eigVec

# concatenate hit, miss, false alarm, correct reject before clustering
layer = 'all'
state = 'active'
t = psthBins-psthBinSize/2
psthAllUnits = {} 
clustData = {}
for region,lbl in zip(('MRN',('SCig','SCiw')),('MRN','SC')):
    psthAllUnits[lbl] = {}
    for resp in respLabels:
        d = []
        for sessionId in sessionIds:
            if (all(len(psth[sessionId][region][layer][state][resp])>0 and
                not np.all(np.isnan(psth[sessionId][region][layer][state][resp]))for resp in respLabels)):
                b = base[sessionId][region][layer][state][resp][0]
                d.append(psth[sessionId][region][layer][state][resp][0] - b[:,None])
        psthAllUnits[lbl][resp] = np.concatenate(d)
    clustData[lbl] = np.concatenate([psthAllUnits[lbl][resp] for resp in respLabels],axis=1)

pcaData,eigVal,eigVec = pca(clustData['SC'],plot=True)

clustId,linkageMat = cluster(pcaData[:,:100],nClusters=5,plot=True,colors=None,labels='off',nreps=10)

clustLabels = np.unique(clustId)

 

## adaptation
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
                    if (not row['omitted'] and not row['previous_omitted']
                        and row['flashes_since_change'] > 5
                        and lastLick < row['flashes_since_change']):
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
for region,clr,lbl in zip(regions,regionColors,regionLabels):
    d = np.concatenate(adaptSpikes[region]['all'])
    d -= d[:,baseWin].mean(axis=1)[:,None]
    d /= binSize
    ax.plot(t,d.mean(axis=0),color=clr,alpha=0.5,label=lbl+', n='+str(d.shape[0]))            
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlim([-0.25,7.5])
ax.set_xlabel('Time from change (s)')
ax.set_ylabel('Spikes/s')
ax.legend(loc='upper center',fontsize=8)
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for region,clr,lbl in zip(regions,regionColors,regionLabels):
    d = np.concatenate(adaptResp[region]['all'])
    d /= d[:,1][:,None]
    d[np.isinf(d)] = np.nan
    mean = np.nanmean(d,axis=0)
    sem = np.nanstd(d,axis=0)/(d.shape[0]**0.5)
    ax.plot(flashTimes,mean,color=clr,alpha=0.5,label=lbl)
    for x,m,s in zip(flashTimes,mean,sem):
        ax.plot([x,x],[m-s,m+s],color=clr,alpha=0.5)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlim([-1,7.5])
ax.set_ylim([0,1.01])
ax.set_xlabel('Time from change (s)')
ax.set_ylabel('Norm. response')
ax.legend(loc='upper center',fontsize=8)
plt.tight_layout()

fig = plt.figure(figsize=(6,11))
for i,region in enumerate(r for r in regions if 'VIS' in r):
    ax = fig.add_subplot(6,1,i+1)
    for layer,clr in zip(layers,layerColors):
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
    ax.legend(loc='upper center',fontsize=8)
    ax.set_title(region)
plt.tight_layout()

fig = plt.figure(figsize=(6,11))
for i,region in enumerate(r for r in regions if 'VIS' in r):
    ax = fig.add_subplot(6,1,i+1)
    for layer,clr in zip(layers,layerColors):
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
        ax.legend(loc='upper center',fontsize=8)
    ax.set_title(region)
plt.tight_layout()

fig = plt.figure(figsize=(10,8))
xticks = np.arange(len(regions))
for i,layer in enumerate(layers):
    ax = fig.add_subplot(5,1,i+1)
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
    ax.set_xticklabels(regionLabels)
    ax.set_ylim([0,1.01])
    ax.set_ylabel('Adaptation ratio')
    ax.set_title('cortical layer '+str(layer))
plt.tight_layout()


flashSinceLickTimes = np.arange(0.75,0.75*13,0.75)
for r,ylbl in zip((flashResp,flashBase,changeFlashResp,changeFlashBase),('flash resp','pre-flash baseline','change resp','pre-change baseline')):
    fig = plt.figure(figsize=(7,11))
    for i,layer in enumerate(layers):
        ax = fig.add_subplot(5,1,i+1)
        ymax = 0
        for region,clr,lbl in zip(regions,regionColors,regionLabels):
            if 'VIS' in region:
                d = np.concatenate(r[region][layer])
            else:
                d = np.concatenate(r[region][layers[0]])
            d /= (respWin.stop-respWin.start)/1000
            mean = np.nanmean(d,axis=0)
            sem = np.nanstd(d,axis=0)/(d.shape[0]**0.5)
            lbl = lbl if i==0 else None
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
            ax.legend(loc=loc,fontsize=5)
        ax.set_title('cortical layer '+str(layer))
    plt.tight_layout()


## change/lick decoding
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


model = LinearSVC(C=1.0,max_iter=1e4)

unitSampleSize = [5,10,15,20,25,30,40,50,60,70,80]

nCrossVal = 5

decodeWindowSize = 10
decodeWindowEnd = 500 # respWin.stop
decodeWindows = np.arange(decodeWindowSize,decodeWindowEnd+decodeWindowSize,decodeWindowSize)

decodeData = {sessionId: {region: {layer: {sampleSize: {} for sampleSize in unitSampleSize} for layer in layers} for region in regions} for sessionId in sessionIds}
for sessionId in sessionIds:
    for region in regions:
        for layer in layers:
            decodeData[sessionId][region][layer]['nUnits'] = 0
            decodeData[sessionId][region][layer]['psth'] = {}
            for lbl in ('hit','miss','false alarm','correct reject'):
                decodeData[sessionId][region][layer]['psth'][lbl] = []

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
    
    engagedChange,engagedCatch = [np.array([np.sum(hit[(changeTimes>t-60) & (changeTimes<t+60)]) > 1 for t in times]) for times in (changeTimes,catchTimes)]
    changeFlash = changeFlash[engagedChange]
    catchFlash = catchFlash[engagedCatch]
    nChangeTrials = engagedChange.sum()
    nCatchTrials = engagedCatch.sum()
    
    hit = hit[engagedChange]
    falseAlarm = np.array(stim['false_alarm'][catchFlash])
    decodeData[sessionId]['changeBehav'] = hit
    decodeData[sessionId]['catchBehav'] = falseAlarm
    
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
                
            changeSp = sp[:,changeFlash,:]
            preChangeSp = sp[:,changeFlash-1,:]
            hasResp = findResponsiveUnits(preChangeSp,changeSp,baseWin,respWin)
            nUnits = hasResp.sum()
            decodeData[sessionId][region][layer]['nUnits'] = nUnits
            if not any(hasResp):
                continue
            
            base = sp[hasResp,:,baseWin].sum(axis=2)
            resp = np.full((hasResp.sum(),len(stim)),np.nan)
            resp[:,1:] = sp[hasResp,1:,respWin].sum(axis=2) - base[:,:-1]
            decodeData[sessionId][region][layer]['changeResp'] = resp[:,changeFlash]
            decodeData[sessionId][region][layer]['preChangeResp'] = resp[:,changeFlash-1]
            
            changeSp,preChangeSp = [s[hasResp,:,:decodeWindows[-1]].reshape((nUnits,nChangeTrials,len(decodeWindows),decodeWindowSize)).sum(axis=-1) for s in (changeSp,preChangeSp)]
            catchSp = sp[hasResp][:,catchFlash,:decodeWindows[-1]].reshape((nUnits,nCatchTrials,len(decodeWindows),decodeWindowSize)).sum(axis=-1)
            
            for s,flash,lick,lbls in zip((changeSp,catchSp),(changeFlash,catchFlash),(hit,falseAlarm),(('hit','miss'),('false alarm','correct reject'))):
                for r,lbl in zip((lick,~lick),lbls): 
                    decodeData[sessionId][region][layer]['psth'][lbl].append(np.mean(s[:,r]/decodeWindowSize*1000 - base[:,flash[r]-1,None]/(baseWin.stop-baseWin.start)/binSize,axis=1))
            
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
                        Ylick = np.concatenate((hit,falseAlarm))
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

#
sampleSize = 20        
winInd = np.where(decodeWindows==respWin.stop)[0][0]

# unit sample size        
fig = plt.figure(figsize=(14,8))       
for i,region in enumerate(regions):
    ax = fig.add_subplot(3,5,i+1)
    y = 1
    for layer,clr in zip(layers,layerColors):
        if layer==layers[0] or 'VIS' in region:
            mean = []
            sem = []
            nUnits = []
            for s in unitSampleSize:
                lyr = layer if 'VIS' in region else layers[0]
                d = [decodeData[sessionId][region][lyr][s]['changeAccuracy'][winInd] for sessionId in sessionIds if len(decodeData[sessionId][region][lyr][s])>0]
                mean.append(np.mean(d))
                sem.append(np.std(d)/(len(d)**0.5))
                nUnits.append(len(d))
            lbl = layer if region=='VISp' else None
            ax.plot(unitSampleSize,mean,'-o',color=clr,mfc='none',label=lbl)
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
    if i==5:
        ax.set_ylabel('Change decoding accuracy')
    if i==12:
        ax.set_xlabel('Number of neurons')
    ax.set_title(region)
    if region=='VISp':
        ax.legend(loc='lower right',fontsize=6)
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for region,clr,lbl in zip(regions,regionColors,regionLabels):
    mean = []
    sem = []
    x = []
    for s in unitSampleSize:
        d = [decodeData[sessionId][region]['all'][s]['changeAccuracy'][winInd] for sessionId in sessionIds if len(decodeData[sessionId][region]['all'][s])>0]
        if len(d)>2:
            mean.append(np.mean(d))
            sem.append(np.std(d)/(len(d)**0.5))
            x.append(s)
    ax.plot(x,mean,color=clr,label=lbl)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlim([0,1.05*max(unitSampleSize)])
ax.set_ylim([0.5,1])
ax.set_xlabel('Number of neurons (from each of at least 3 sessions)')
ax.set_ylabel('Change decoding accuracy')
ax.legend(loc='upper left',fontsize=6)
plt.tight_layout()

# change decoding
fig = plt.figure(figsize=(10,8))
for i,layer in enumerate(layers):
    ax = fig.add_subplot(3,2,i+1)
    for region,clr in zip(regions,regionColors):
        lyr = layer if 'VIS' in region else layers[0]
        d = [decodeData[sessionId][region][lyr][sampleSize]['changeAccuracy'] for sessionId in sessionIds if len(decodeData[sessionId][region][lyr][sampleSize])>0]
        if len(d)>0:
            m = np.mean(d,axis=0)
            s = np.std(d,axis=0)/(len(d)**0.5)
            ax.plot(decodeWindows-decodeWindowSize/2,m,color=clr)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_ylim([0.4,1])
    ax.set_xlabel('Time from change (ms)')
    ax.set_ylabel('Change decoding accuracy')
    ax.set_title('cortical layer '+str(layer))
ax = fig.add_subplot(3,2,6)
for lbl,clr in zip(regionLabels,regionColors):
    ax.plot([],color=clr,label=lbl)
for side in ('right','top','left','bottom'):
    ax.spines[side].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])
ax.legend(loc='center',fontsize=8)
plt.tight_layout()

fig = plt.figure(figsize=(12,8))
xticks = np.arange(len(regions))
for i,layer in enumerate(layers):
    ax = fig.add_subplot(3,2,i+1)
    for x,region in enumerate(regions):
        lyr = layer if 'VIS' in region else layers[0]
        for winEnd,mfc in zip((100,200),('k','none')):
            j = np.where(decodeWindows==winEnd)[0][0]
            d = [decodeData[sessionId][region][lyr][sampleSize]['changeAccuracy'][j] for sessionId in sessionIds if len(decodeData[sessionId][region][lyr][sampleSize])>0]
            m = np.mean(d)
            s = np.std(d)/(len(d)**0.5)
            lbl = str(winEnd)+' ms' if region=='VISp' else None
            ax.plot(x,m,'ko',mfc=mfc,label=lbl)
            ax.plot([x,x],[m-s,m+s],'k')
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xticks(xticks)
    ax.set_xticklabels(regionLabels)
    ax.set_ylim([0.5,1])
    ax.set_ylabel('Change decoding accuracy')
    ax.set_title('cortical layer '+str(layer))
    if i==0:
        ax.legend(loc='upper left')
plt.tight_layout()
    
fig = plt.figure(figsize=(10,8))
for i,layer in enumerate(layers):
    ax = fig.add_subplot(3,2,i+1)
    for region,clr in zip(regions,regionColors): 
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
    if i==2:
        ax.set_ylabel('Correlation of decoder confidence and behavior')
    ax.set_title('cortical layer '+str(layer))
ax = fig.add_subplot(3,2,6)
for lbl,clr in zip(regionLabels,regionColors):
    ax.plot([],color=clr,label=lbl)
for side in ('right','top','left','bottom'):
    ax.spines[side].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])
ax.legend(loc='center',fontsize=8)
plt.tight_layout()

fig = plt.figure(figsize=(12,8))
for i,layer in enumerate(layers):
    ax = fig.add_subplot(3,2,i+1)
    for x,region in enumerate(regions):
        lyr = layer if 'VIS' in region else layers[0]
        for winEnd,mfc in zip((100,200),('k','none')):
            j = np.where(decodeWindows==winEnd)[0][0]
            r = []
            for sessionId in sessionIds:
                if len(decodeData[sessionId][region][lyr][sampleSize])>0:
                    b = np.concatenate([decodeData[sessionId][resp] for resp in ('changeBehav','catchBehav')])
                    d = np.concatenate([decodeData[sessionId][region][lyr][sampleSize][conf][j] for conf in ('changeConfidence','catchConfidence')])
                    r.append(np.corrcoef(b,d)[0,1])
            m = np.mean(r)
            s = np.std(r)/(len(r)**0.5)
            lbl = str(winEnd)+' ms' if region=='VISp' else None
            ax.plot(x,m,'ko',mfc=mfc,label=lbl)
            ax.plot([x,x],[m-s,m+s],'k')
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xticks(xticks)
    ax.set_xticklabels(regionLabels)
    ax.set_ylim([0,0.8])
    if i==2:
        ax.set_ylabel('Correlation of decoder confidence and behavior')
    ax.set_title('cortical layer '+str(layer))
    if i==0:
        ax.legend(loc='upper left')
plt.tight_layout()
    
fig = plt.figure(figsize=(12,8))
for i,layer in enumerate(layers):
    ax = fig.add_subplot(3,2,i+1)
    for x,region in enumerate(regions):
        lyr = layer if 'VIS' in region else layers[0]
        for winEnd,mfc in zip((100,200),('k','none')):
            j = np.where(decodeWindows==winEnd)[0][0]
            js = []
            for sessionId in sessionIds:
                if len(decodeData[sessionId][region][lyr][sampleSize])>0:
                    b = np.concatenate([decodeData[sessionId][resp] for resp in ('changeBehav','catchBehav')])
                    d = np.concatenate([decodeData[sessionId][region][lyr][sampleSize][pred][j] for pred in ('changePrediction','catchPrediction')]).astype(bool)
                    js.append((b & d).sum() / (b | d).sum())
            m = np.mean(js)
            s = np.std(js)/(len(js)**0.5)
            lbl = str(winEnd)+' ms' if region=='VISp' else None
            ax.plot(x,m,'ko',mfc=mfc,label=lbl)
            ax.plot([x,x],[m-s,m+s],'k')
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xticks(xticks)
    ax.set_xticklabels(regionLabels)
    ax.set_ylim([0.4,0.9])
    if i==2:
        ax.set_ylabel('Jaccard similarity between decoder and behavior')
    ax.set_title('cortical layer '+str(layer))
    if i==0:
        ax.legend(loc='upper left')
plt.tight_layout()

# change decoding feature weights
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for region,clr,lbl in zip(regions,regionColors,regionLabels):
    d = [decodeData[sessionId][region][layer][sampleSize]['changeFeatureWeights'][-1] for sessionId in sessionIds for layer in layers if len(decodeData[sessionId][region][layer][sampleSize])>0]
    if len(d)>0:
        d = np.concatenate(d)
        m = np.nanmean(d,axis=0)
        s = np.std(d,axis=0)/(len(d)**0.5)
        ax.plot(decodeWindows-decodeWindowSize/2,m,color=clr,label=lbl)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlabel('Time from change (ms)')
ax.set_ylabel('Change decoder weighting')
ax.legend(loc='upper right',fontsize=8)
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

# d' 
fig = plt.figure(figsize=(10,8))
alim = [-0.5,4]  
for i,layer in enumerate(layers):
    ax = fig.add_subplot(3,2,i+1)
    ax.plot(alim,alim,'--',color='0.75')
    for region,clr,lbl in zip(regions,regionColors,regionLabels):
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
            ax.plot(d[:,0],d[:,1],'o',mec=clr,mfc='none',label=lbl)
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
    ax.set_title('cortical layer '+str(layer))
ax = fig.add_subplot(3,2,6)
for lbl,clr in zip(regionLabels,regionColors):
    ax.plot([],color=clr,label=lbl)
for side in ('right','top','left','bottom'):
    ax.spines[side].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])
ax.legend(loc='center',fontsize=8)
plt.tight_layout()

# lick decoding
fig = plt.figure(figsize=(10,8))
for i,layer in enumerate(layers):
    ax = fig.add_subplot(3,2,i+1)
    for region,clr in zip(regions,regionColors):
        lyr = layer if 'VIS' in region else layers[0]
        d = [decodeData[sessionId][region][lyr][sampleSize]['lickBalancedAccuracy'] for sessionId in sessionIds if len(decodeData[sessionId][region][lyr][sampleSize])>0]
        if len(d)>0:
            m = np.mean(d,axis=0)
            s = np.std(d,axis=0)/(len(d)**0.5)
            ax.plot(decodeWindows-decodeWindowSize/2,m,color=clr,label=region)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_ylim([0.4,1])
    ax.set_xlabel('Time from change (ms)')
    ax.set_ylabel('Lick decoding balanced accuracy')
    ax.set_title('cortical layer '+str(layer))
ax = fig.add_subplot(3,2,6)
for lbl,clr in zip(regionLabels,regionColors):
    ax.plot([],color=clr,label=lbl)
for side in ('right','top','left','bottom'):
    ax.spines[side].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])
ax.legend(loc='center',fontsize=8)
plt.tight_layout()

fig = plt.figure(figsize=(12,8))
for i,layer in enumerate(layers):
    ax = fig.add_subplot(3,2,i+1)
    for x,region in enumerate(regions):
        lyr = layer if 'VIS' in region else layers[0]
        for winEnd,mfc in zip((100,200),('k','none')):
            j = np.where(decodeWindows==winEnd)[0][0]
            d = [decodeData[sessionId][region][lyr][sampleSize]['lickBalancedAccuracy'][j] for sessionId in sessionIds if len(decodeData[sessionId][region][lyr][sampleSize])>0]
            if len(d)>0:
                m = np.mean(d)
                s = np.std(d)/(len(d)**0.5)
                lbl = str(winEnd)+' ms' if region=='VISp' else None
                ax.plot(x,m,'ko',mfc=mfc,label=lbl)
                ax.plot([x,x],[m-s,m+s],'k')
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xticks(xticks)
    ax.set_xticklabels(regionLabels)
    ax.set_ylim([0.45,1])
    if i==2:
        ax.set_ylabel('Lick decoding balanced accuracy')
    ax.set_title('cortical layer '+str(layer))
    if i==0:
        ax.legend(loc='upper left')
plt.tight_layout()

layer = 'all'
x = decodeWindows-decodeWindowSize/2      
for i,(region,lbl) in enumerate(zip(regions,regionLabels)):
    if lbl not in ('SC','MRN'):
        continue
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for resp,clr in zip(('hit','miss','false alarm','correct reject'),'krgb'):
        d = []
        for sessionId in sessionIds:
            if len(decodeData[sessionId][region][layer][sampleSize])>0:
                changeBehav = decodeData[sessionId]['changeBehav']
                catchBehav = decodeData[sessionId]['catchBehav']
                if resp=='hit':
                    trials = np.concatenate((changeBehav,np.zeros_like(catchBehav)))
                elif resp=='miss':
                    trials = np.concatenate((~changeBehav,np.zeros_like(catchBehav)))
                elif resp=='false alarm':
                    trials = np.concatenate((np.zeros_like(changeBehav),catchBehav))
                elif resp=='correct reject':
                    trials = np.concatenate((np.zeros_like(changeBehav),~catchBehav))
                d.append(np.mean(decodeData[sessionId][region][layer][sampleSize]['lickConfidence'][:,trials],axis=1))
        m = np.nanmean(d,axis=0)
        s = np.nanstd(d,axis=0)/(len(d)**0.5)
        ax.plot(x,m,color=clr,label=resp)
        ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xlabel('Time from change/catch (ms)')
    ax.set_ylabel('Lick decoder confidence')
    ax.legend()
    ax.set_title(region)
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for region,clr,lbl in zip(regions,regionColors,regionLabels):
    d = [decodeData[sessionId][region][layer][sampleSize]['lickFeatureWeights'][-1] for sessionId in sessionIds for layer in layers if len(decodeData[sessionId][region][layer][sampleSize])>0]
    if len(d)>0:
        d = np.concatenate(d)
        m = np.nanmean(d,axis=0)
        s = np.std(d,axis=0)/(len(d)**0.5)
        ax.plot(decodeWindows-decodeWindowSize/2,m,color=clr,label=lbl)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlabel('Time from change (ms)')
ax.set_ylabel('Lick decoder weighting')
ax.legend(loc='upper right',fontsize=8)
plt.tight_layout()

# psth
fig = plt.figure(figsize=(16,8))
layer = 'all' 
x = decodeWindows-decodeWindowSize/2      
for i,region,lbl in enumerate(zip(regions,regionLabels)):
    ax = fig.add_subplot(3,5,i+1)
    for resp,clr in zip(('hit','miss','false alarm','correct reject'),'krgb'):
        d = np.concatenate([decodeData[sessionId][region][layer]['psth'][resp][0] for sessionId in sessionIds if len(decodeData[sessionId][region][layer]['psth'][resp])>0])
        m = np.nanmean(d,axis=0)
        s = np.nanstd(d,axis=0)/(len(d)**0.5)
        ax.plot(x,m,color=clr,label=resp)
        ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xlabel('Time from change/catch (ms)')
    ax.set_ylabel('Spikes/s')
    if i==0:
        ax.legend()
    ax.set_title(lbl)
plt.tight_layout()

# change decoding vs behavior correlation without catch trials
fig = plt.figure(figsize=(10,8))
for i,layer in enumerate(layers):
    ax = fig.add_subplot(3,2,i+1)
    for region,clr in zip(regions,regionColors): 
        lyr = layer if 'VIS' in region else layers[0]
        r = []
        for sessionId in sessionIds:
            if len(decodeData[sessionId][region][lyr][sampleSize])>0:
                b = decodeData[sessionId]['changeBehav']
                r.append([])
                for j,_ in enumerate(decodeWindows):
                    d = decodeData[sessionId][region][lyr][sampleSize]['changeConfidence'][j]
                    r[-1].append(np.corrcoef(b,d)[0,1])
        if len(r)>0:
            m = np.nanmean(r,axis=0)
            s = np.nanstd(r,axis=0)/(len(r)**0.5)
            ax.plot(decodeWindows-decodeWindowSize/2,m,color=clr,label=region)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_ylim([-0.1,0.8])
    ax.set_xlabel('Time from change (ms)')
    if i==2:
        ax.set_ylabel('Correlation of decoder confidence and behavior')
    ax.set_title('cortical layer '+str(layer))
ax = fig.add_subplot(3,2,6)
for lbl,clr in zip(regionLabels,regionColors):
    ax.plot([],color=clr,label=lbl)
for side in ('right','top','left','bottom'):
    ax.spines[side].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])
ax.legend(loc='center',fontsize=8)
plt.tight_layout()

fig = plt.figure(figsize=(12,8))
for i,layer in enumerate(layers):
    ax = fig.add_subplot(3,2,i+1)
    for x,region in enumerate(regions):
        lyr = layer if 'VIS' in region else layers[0]
        for winEnd,mfc in zip((100,200),('k','none')):
            j = np.where(decodeWindows==winEnd)[0][0]
            r = []
            for sessionId in sessionIds:
                if len(decodeData[sessionId][region][lyr][sampleSize])>0:
                    b = decodeData[sessionId]['changeBehav']
                    d = decodeData[sessionId][region][lyr][sampleSize]['changeConfidence'][j]
                    r.append(np.corrcoef(b,d)[0,1])
            m = np.nanmean(r)
            s = np.nanstd(r)/(len(r)**0.5)
            lbl = str(winEnd)+' ms' if region=='VISp' else None
            ax.plot(x,m,'ko',mfc=mfc,label=lbl)
            ax.plot([x,x],[m-s,m+s],'k')
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xticks(xticks)
    ax.set_xticklabels(regionLabels)
    ax.set_ylim([-0.1,0.6])
    if i==2:
        ax.set_ylabel('Correlation of decoder confidence and behavior')
    ax.set_title('cortical layer '+str(layer))
    if i==0:
        ax.legend(loc='upper left')
plt.tight_layout()


# non-change lick decoding
lickDecodeData = {sessionId: {region: {layer: {} for layer in layers} for region in regions} for sessionId in sessionIds}
sampleSize = 20
decodeWindowEnd = 500
decodeWindows = np.arange(decodeWindowSize,decodeWindowEnd+decodeWindowSize,decodeWindowSize)

warnings.filterwarnings('ignore')
for sessionIndex,sessionId in enumerate(sessionIds):
    units = unitTable.set_index('unit_id').loc[unitData[str(sessionId)]['unitIds'][:]]
    spikes = unitData[str(sessionId)]['spikes']
    stim = stimTable[(stimTable['session_id']==sessionId) & stimTable['active']].reset_index()
    flashTimes = np.array(stim['start_time'])
    changeTimes = flashTimes[stim['is_change']]
    hit = np.array(stim['hit'][stim['is_change']])
    engaged = np.array([np.sum(hit[(changeTimes>t-60) & (changeTimes<t+60)]) > 1 for t in flashTimes])
    autoRewarded = np.array(stim['auto_rewarded']).astype(bool)
    changeFlash = np.where(stim['is_change'] & ~autoRewarded & engaged)[0]
    nonChangeFlashes = np.array(engaged &
                                (~stim['is_change']) & 
                                (~stim['omitted']) & 
                                (~stim['previous_omitted']) & 
                                (stim['flashes_since_change']>5) &
                                (stim['flashes_since_last_lick']>1))
    lick = np.array(stim['lick_for_flash'])[nonChangeFlashes]
    
    lickDecodeData[sessionId]['image'] = np.array(stim['image_name'])[nonChangeFlashes]
    lickDecodeData[sessionId]['flashesSinceLick'] = np.array(stim['flashes_since_last_lick'])[nonChangeFlashes]
    lickDecodeData[sessionId]['lick'] = lick
    
    nFlashes = nonChangeFlashes.sum()
    for region in regions:
        inRegion = np.in1d(units['structure_acronym'],region)
        if not any(inRegion):
            continue
        for layer in ('all',):
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
                
            changeSp = sp[:,changeFlash,:]
            preChangeSp = sp[:,changeFlash-1,:]
            hasResp = findResponsiveUnits(preChangeSp,changeSp,baseWin,respWin)
            nUnits = hasResp.sum()
            if nUnits < sampleSize:
                continue
            flashSp = sp[hasResp][:,nonChangeFlashes,:decodeWindows[-1]].reshape((nUnits,nFlashes,len(decodeWindows),decodeWindowSize)).sum(axis=-1)
            
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
            trainAccuracy = np.full((len(unitSamples),len(decodeWindows)),np.nan)
            featureWeights = np.full((len(unitSamples),len(decodeWindows),nUnits,len(decodeWindows)),np.nan)
            accuracy = trainAccuracy.copy()
            balancedAccuracy = accuracy.copy()
            prediction = np.full((len(unitSamples),len(decodeWindows),nFlashes),np.nan)
            confidence = prediction.copy()
            for i,unitSamp in enumerate(unitSamples):
                for j,winEnd in enumerate((decodeWindows/decodeWindowSize).astype(int)):
                    X = flashSp[unitSamp,:,:winEnd].transpose(1,0,2).reshape((nFlashes,-1))                        
                    cv = crossValidate(model,X,lick,nCrossVal)
                    trainAccuracy[i,j] = np.mean(cv['train_score'])
                    featureWeights[i,j,unitSamp,:winEnd] = np.mean(cv['coef'],axis=0).reshape(sampleSize,winEnd)
                    accuracy[i,j] = np.mean(cv['test_score'])
                    balancedAccuracy[i,j] = sklearn.metrics.balanced_accuracy_score(lick,cv['predict'])
                    prediction[i,j] = cv['predict']
                    confidence[i,j] = cv['decision_function']
            lickDecodeData[sessionId][region][layer]['trainAccuracy'] = np.median(trainAccuracy,axis=0)
            lickDecodeData[sessionId][region][layer]['featureWeights'] = np.nanmedian(featureWeights,axis=0)
            lickDecodeData[sessionId][region][layer]['accuracy'] = np.median(accuracy,axis=0)
            lickDecodeData[sessionId][region][layer]['balancedAccuracy'] = np.median(balancedAccuracy,axis=0) 
            lickDecodeData[sessionId][region][layer]['prediction'] = scipy.stats.mode(prediction,axis=0)[0][0]
            lickDecodeData[sessionId][region][layer]['confidence'] = np.median(confidence,axis=0)
warnings.filterwarnings('default')


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
lyr = 'all'
for region,clr,lbl in zip(regions,regionColors,regionLabels):
    d = [lickDecodeData[sessionId][region][lyr]['balancedAccuracy'] for sessionId in sessionIds if len(lickDecodeData[sessionId][region][lyr])>0]
    if len(d)>0:
        m = np.mean(d,axis=0)
        s = np.std(d,axis=0)/(len(d)**0.5)
        ax.plot(decodeWindows-decodeWindowSize/2,m,color=clr,label=lbl)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_ylim([0.45,0.85])
ax.set_xlabel('Time from non-change flash onset (ms)')
ax.set_ylabel('Lick decoding balanced accuracy')
ax.legend(loc='upper left',fontsize=8)
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for x,region in enumerate(regions):
    for winEnd,mfc in zip((100,200),('k','none')):
        j = np.where(decodeWindows==winEnd)[0][0]
        d = [lickDecodeData[sessionId][region][lyr]['balancedAccuracy'][j] for sessionId in sessionIds if len(lickDecodeData[sessionId][region][lyr])>0]
        if len(d)>0:
            m = np.nanmean(d)
            s = np.nanstd(d)/(len(d)**0.5)
            lbl = str(winEnd)+' ms' if region=='VISp' else None
            ax.plot(x,m,'ko',mfc=mfc,label=lbl)
            ax.plot([x,x],[m-s,m+s],'k')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xticks(np.arange(len(regions)))
ax.set_xticklabels(regionLabels)
ax.set_ylim([0.45,0.85])
ax.set_ylabel('Lick decoding balanced accuracy')
ax.legend(loc='upper left')
plt.tight_layout()

x = decodeWindows-decodeWindowSize/2      
for i,(region,lbl) in enumerate(zip(regions,regionLabels)):
    if lbl not in ('SC','MRN'):
        continue
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for resp,clr in zip(('lick','no lick'),'gm'):
        d = []
        for sessionId in sessionIds:
            if len(lickDecodeData[sessionId][region]['all'])>0:
                lick = lickDecodeData[sessionId]['lick']
                if resp=='no lick':
                    lick = ~lick
                d.append(np.mean(lickDecodeData[sessionId][region]['all']['confidence'][:,lick],axis=1))
        m = np.nanmean(d,axis=0)
        s = np.nanstd(d,axis=0)/(len(d)**0.5)
        ax.plot(x,m,color=clr,label=resp)
        ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xlabel('Time from non-change flash onset (ms)')
    ax.set_ylabel('Lick decoder confidence')
    ax.legend()
    ax.set_title(region)
plt.tight_layout()

flashSinceLick = np.arange(2,13)
lickProb = []
for i,(region,lbl) in enumerate(zip(regions,regionLabels)):
    if lbl not in ('VISrl',):#'SC','MRN','VISp'):
        continue
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for f,clr in zip(flashSinceLick,plt.cm.magma(np.linspace(0,0.9,len(flashSinceLick)))):
        d = []
        p = []
        for sessionId in sessionIds:
            if len(lickDecodeData[sessionId][region]['all'])>0:
                trials = lickDecodeData[sessionId]['flashesSinceLick']==f
                lick = lickDecodeData[sessionId]['lick'][trials]
                d.append([])
                for j,_ in enumerate(decodeWindows):
                    pred = lickDecodeData[sessionId][region][layer]['prediction'][j,trials]
                    d[-1].append(sklearn.metrics.balanced_accuracy_score(lick,pred))
        m = np.nanmean(d,axis=0)
        s = np.nanstd(d,axis=0)/(len(d)**0.5)
        ax.plot(x,m,color=clr,label=f)
        ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    # ax.set_ylim([0.45,1])
    ax.set_xlabel('Time from non-change flash onset (ms)')
    ax.set_ylabel('Lick decoder balanced accuracy')
    ax.legend(loc='upper left',title='flashes since last lick',fontsize=8)
    ax.set_title(region)
plt.tight_layout()

imgs = np.unique([np.unique(lickDecodeData[sessionId]['image']) for sessionId in sessionIds])
for i,(region,lbl) in enumerate(zip(regions,regionLabels)):
    if lbl not in ('SC','MRN'):
        continue
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for img,clr in zip(imgs,plt.cm.tab20(np.linspace(0,1,len(imgs)))):
        d = []
        for sessionId in sessionIds:
            if len(lickDecodeData[sessionId][region]['all'])>0:
                trials = lickDecodeData[sessionId]['image']==img
                lick = lickDecodeData[sessionId]['lick'][trials]
                d.append([])
                for j,_ in enumerate(decodeWindows):
                    pred = lickDecodeData[sessionId][region][layer]['prediction'][j,trials]
                    d[-1].append(sklearn.metrics.balanced_accuracy_score(lick,pred))
        m = np.nanmean(d,axis=0)
        s = np.nanstd(d,axis=0)/(len(d)**0.5)
        ax.plot(x,m,color=clr,label=img)
        ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xlabel('Time from non-change flash onset (ms)')
    ax.set_ylabel('Lick decoder balanced accuracy')
    ax.legend(loc='upper left',fontsize=8)
    ax.set_title(region)
plt.tight_layout()









