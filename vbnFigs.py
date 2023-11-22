# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 10:06:06 2023

@author: svc_ccg
"""

import copy
import glob
import os
import numpy as np
import pandas as pd
import h5py
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
import sklearn.metrics
from vbnAnalysisUtils import getBehavData, getUnitsInRegion, findResponsiveUnits, cluster


baseDir = r'\\allen\programs\mindscope\workgroups\np-behavior\vbn_data_release\supplemental_tables'

outputDir = r'\\allen\programs\mindscope\workgroups\np-behavior\VBN_video_analysis'

stimTable = pd.read_csv(os.path.join(baseDir,'master_stim_table.csv'))

unitTable = pd.read_csv(os.path.join(baseDir,'units_with_cortical_layers.csv'))
unitData = h5py.File(os.path.join(baseDir,'vbnAllUnitSpikeTensor.hdf5'),mode='r')

sessionIds = stimTable['session_id'].unique()


# number of flashes and licks
nFlashes = []
nLicks = []
for sessionId in sessionIds:
    stim = stimTable[(stimTable['session_id']==sessionId) & stimTable['active']].reset_index()
    flashTimes,changeFlashes,catchFlashes,nonChangeFlashes,omittedFlashes,prevOmittedFlashes,novelFlashes,lick,lickTimes = getBehavData(stim)
    nFlashes.append(nonChangeFlashes.sum())
    nLicks.append(lick[nonChangeFlashes].sum())
nFlashes = np.array(nFlashes)
nLicks = np.array(nLicks)
lickProb = nLicks / nFlashes

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
binWidth = 100
bins = np.arange(0,nFlashes.max()+binWidth/2,binWidth)
h = np.histogram(nFlashes,bins)[0]
ax.bar(bins[:-1]+binWidth/2,h,width=binWidth,color='k',edgecolor='k')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlabel('Number of flashes')
ax.set_ylabel('Number of sessions')
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
binWidth = 10
bins = np.arange(0,nLicks.max()+binWidth/2,binWidth)
h = np.histogram(nLicks,bins)[0]
ax.bar(bins[:-1]+binWidth/2,h,width=binWidth,color='k',edgecolor='k')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlabel('Number of licks')
ax.set_ylabel('Number of sessions')
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
binWidth = 0.01
bins = np.arange(0,lickProb.max()+binWidth/2,binWidth)
h = np.histogram(lickProb,bins)[0]
ax.bar(bins[:-1]+binWidth/2,h,width=binWidth,color='k',edgecolor='k')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlabel('Fraction of flashes with lick')
ax.set_ylabel('Number of sessions')
plt.tight_layout()


# lick latency
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
lickLatBins = np.arange(0,0.751,1/60)
lickLatTime = lickLatBins[:-1]+0.5/60
cumProbLick = []
for sessionId in sessionIds:
    stim = stimTable[(stimTable['session_id']==sessionId) & stimTable['active']].reset_index()
    flashTimes,changeFlashes,catchFlashes,nonChangeFlashes,omittedFlashes,prevOmittedFlashes,novelFlashes,lick,lickTimes = getBehavData(stim)
    if lick[nonChangeFlashes].sum() >= 10:
        lickLatency = (lickTimes - flashTimes)[nonChangeFlashes & lick]
        dsort = np.sort(lickLatency)
        cumProb = np.array([np.sum(dsort<=i)/dsort.size for i in dsort])
        ax.plot(dsort,cumProb,'0.5',alpha=0.25)
        h = np.histogram(lickLatency,lickLatBins)[0]
        cumProbLick.append(np.cumsum(h)/np.sum(h))
m = np.mean(cumProbLick,axis=0)
s = np.std(cumProbLick,axis=0)/(len(cumProbLick)**0.5)
ax.plot(lickLatTime,m,color='k',lw=2)
ax.fill_between(lickLatTime,m+s,m-s,color='k',alpha=0.25)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlim([0,0.75])
ax.set_ylim([0,1.01])
ax.set_xlabel('Lick latency (s)')
ax.set_ylabel('Cumulative probability')
plt.tight_layout()
    

# average frame and weighted mask
for key in ('avgframe','motMask'):
    fig = plt.figure(figsize=(6,6))
    gs = matplotlib.gridspec.GridSpec(11,10)
    i = 0
    j = 0
    for k,f in enumerate(glob.glob(os.path.join(outputDir,'facemapData','*.hdf5'))):
        print(k)
        ax = fig.add_subplot(gs[i,j])
        with h5py.File(f,'r') as d:
            if key=='motMask':
                img = np.average(d['motMask'][()],weights=d['motSv'][()],axis=2)
                cmax = np.max(np.absolute(img))
                ax.imshow(img,cmap='bwr',clim=[-cmax,cmax])
            else:
                x = d['xrange_bin'][()]
                y = d['yrange_bin'][()]
                img = d[key][()][y,:][:,x]
                ax.imshow(img,cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        if j == 9:
            i += 1
            j = 0
        else:
            j += 1
    plt.tight_layout()


# facemap lick decoding
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
facemapLickDecoding = []
for i,f in enumerate(glob.glob(os.path.join(outputDir,'facemapLickDecoding','unbalanced','facemapLickDecoding_*.npy'))):
    print(i)
    d = np.load(f,allow_pickle=True).item()
    if d['lick'].sum() >= 10:
        ax.plot(d['decodeWindows'],d['balancedAccuracy'],'0.5',alpha=0.25)
        facemapLickDecoding.append(d['balancedAccuracy'])
m = np.mean(facemapLickDecoding,axis=0)
s = np.std(facemapLickDecoding,axis=0)/(len(facemapLickDecoding)**0.5)
facemapDecodingTime= d['decodeWindows']
ax.plot(facemapDecodingTime,m,color='g',lw=2)
ax.fill_between(facemapDecodingTime,m+s,m-s,color='g',alpha=0.25)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_ylim([0.45,1])
ax.set_xlabel('Time from non-change flash onset (ms)')
ax.set_ylabel('Lick decoding balanced accuracy')
plt.tight_layout()


# unit lick decoding
regions = ('LGd','VISp','VISl','VISrl','VISal','VISpm','VISam','LP',
           'Hipp','Sub','APN','NOT','SC','MRN','MB',
           'SC/MRN cluster 1','SC/MRN cluster 2')
regionColors = plt.cm.tab20(np.linspace(0,1,len(regions)))

unitSampleSize = np.array([1,5,10,15,20,25,30,40,50,60])
decodeWindowSampleSize = 20

unitLickDecoding = {region: [] for region in regions}
lickDecodingSampleSize = {region: [] for region in regions}
for i,f in enumerate(glob.glob(os.path.join(outputDir,'unitLickDecoding','old','unitLickDecoding_*.npy'))):
    print(i)
    d = np.load(f,allow_pickle=True).item()
    if d['lick'].sum() >= 10:
        for region in regions:
            a = d[region][decodeWindowSampleSize]['balancedAccuracy']
            if len(a)>0:
                unitLickDecoding[region].append(a)
                
            lickDecodingSampleSize[region].append([])
            for sampleSize in unitSampleSize:
                a = d[region][sampleSize]['balancedAccuracy']
                if len(a) > 0:
                    lickDecodingSampleSize[region][-1].append(a[-1])
                else:
                    lickDecodingSampleSize[region][-1].append(np.nan)
unitDecodingTime = (d['decodeWindows'] - d['decodeWindows'][0]/2) / 1000
            
fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(1,1,1)
for region,clr in zip(regions,regionColors):
    if len(unitLickDecoding[region])>0:
        m = np.mean(unitLickDecoding[region],axis=0)
        s = np.std(unitLickDecoding[region],axis=0)/(len(unitLickDecoding[region])**0.5)
        ax.plot(unitDecodingTime,m,color=clr,label=region+' (n='+str(len(unitLickDecoding[region]))+')')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_ylim([0.45,1])
ax.set_xlabel('Time from non-change flash onset (s)')
ax.set_ylabel('Lick decoding balanced accuracy')
ax.legend(fontsize=8,bbox_to_anchor=(1,1),loc='upper left')
plt.tight_layout()

fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(1,1,1)
for region,clr in zip(regions,regionColors):
    if len(lickDecodingSampleSize[region])>0:
        m = np.nanmean(lickDecodingSampleSize[region],axis=0)
        n = np.sum(~np.isnan(lickDecodingSampleSize[region]),axis=0)
        s = np.nanstd(lickDecodingSampleSize[region],axis=0)/(n**0.5)
        i = n>2
        ax.plot(unitSampleSize[i],m[i],color=clr,label=region+' (n='+str(len(unitLickDecoding[region]))+')')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_ylim([0.45,1])
ax.set_xlabel('Number of units')
ax.set_ylabel('Lick decoding balanced accuracy')
ax.legend(fontsize=8,bbox_to_anchor=(1,1),loc='upper left')
plt.tight_layout()


# unit change decoding
unitChangeDecoding = {region: [] for region in regions}
changeDecodingSampleSize = {region: [] for region in regions}
changeDecodingCorr = {region: [] for region in regions}
for i,f in enumerate(glob.glob(os.path.join(outputDir,'unitChangeDecoding','old','unitChangeDecoding_*.npy'))):
    print(i)
    d = np.load(f,allow_pickle=True).item()
    hit = d['hit']
    if hit.sum() >= 10:
        for region in regions:
            decodeWindowSampleSize = 20 if region=='SC/MRN cluster 1' else 20
            a = d[region][decodeWindowSampleSize]['accuracy']
            if len(a)>0:
                unitChangeDecoding[region].append(a)
                
            changeDecodingSampleSize[region].append([])
            for sampleSize in unitSampleSize:
                a = d[region][sampleSize]['accuracy']
                if len(a) > 0:
                    changeDecodingSampleSize[region][-1].append(a[-1])
                else:
                    changeDecodingSampleSize[region][-1].append(np.nan)
            
            if hit.sum() < hit.size:
                conf = d[region][decodeWindowSampleSize]['confidence']
                if len(conf)>0:
                    changeDecodingCorr[region].append([np.corrcoef(hit,c[:hit.size])[0,1] for c in conf])

fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(1,1,1)
for region,clr in zip(regions,regionColors):
    if len(unitChangeDecoding[region])>0:
        m = np.mean(unitChangeDecoding[region],axis=0)
        s = np.std(unitChangeDecoding[region],axis=0)/(len(unitChangeDecoding[region])**0.5)
        ax.plot(unitDecodingTime,m,color=clr,label=region+' (n='+str(len(unitChangeDecoding[region]))+')')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_ylim([0.45,1])
ax.set_xlabel('Time from change (s)')
ax.set_ylabel('Change decoding accuracy')
ax.legend(fontsize=8,bbox_to_anchor=(1,1),loc='upper left')
plt.tight_layout()

fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(1,1,1)
for region,clr in zip(regions,regionColors):
    if len(changeDecodingSampleSize[region])>0:
        m = np.nanmean(changeDecodingSampleSize[region],axis=0)
        n = np.sum(~np.isnan(changeDecodingSampleSize[region]),axis=0)
        s = np.nanstd(changeDecodingSampleSize[region],axis=0)/(n**0.5)
        i = n>2
        ax.plot(unitSampleSize[i],m[i],color=clr,label=region+' (n='+str(len(unitChangeDecoding[region]))+')')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_ylim([0.45,1])
ax.set_xlabel('Number of units')
ax.set_ylabel('Change decoding accuracy')
ax.legend(fontsize=8,bbox_to_anchor=(1,1),loc='upper left')
plt.tight_layout()

fig = plt.figure(figsize=(8,5))
ax = fig.add_subplot(1,1,1)
for region,clr in zip(regions,regionColors):
    if len(changeDecodingCorr[region])>0:
        m = np.mean(changeDecodingCorr[region],axis=0)
        s = np.std(changeDecodingCorr[region],axis=0)/(len(changeDecodingCorr[region])**0.5)
        ax.plot(unitDecodingTime,m,color=clr,label=region+' (n='+str(len(changeDecodingCorr[region]))+')')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_ylim([-0.1,1])
ax.set_xlabel('Time from change (s)')
ax.set_ylabel('Change decoder correlation with behavior')
ax.legend(fontsize=8,bbox_to_anchor=(1,1),loc='upper left')
plt.tight_layout()


# psth
regions = ('VISall','VISp','SC','MRN')

baseWin = slice(680,750)
respWin = slice(30,100)

psthBinSize = 5
psthTime = np.arange(0,750,psthBinSize)/1000
psth = {region: {lbl: [] for lbl in ('change','hit','miss','non-change lick','non-change no lick')} for region in regions}
psthUnitId = {region: [] for region in regions}
for sessionInd,sessionId in enumerate(sessionIds):
    print(sessionInd)
    units = unitTable.set_index('unit_id').loc[unitData[str(sessionId)]['unitIds'][:]]
    spikes = unitData[str(sessionId)]['spikes']

    stim = stimTable[(stimTable['session_id']==sessionId) & stimTable['active']].reset_index()
    flashTimes,changeFlashes,catchFlashes,nonChangeFlashes,omittedFlashes,prevOmittedFlashes,novelFlashes,lick,lickTimes = getBehavData(stim)
    outcome = []
    for lbl in ('hit','miss'):
        a = stim[lbl].copy()
        a[a.isnull()] = False
        outcome.append(np.array(a).astype(bool))
    hit = outcome[0] & changeFlashes
    miss = outcome[1] & changeFlashes
    
    for region in regions:
        inRegion = getUnitsInRegion(units,region)
        if not any(inRegion):
            continue
                
        sp = np.zeros((inRegion.sum(),spikes.shape[1],spikes.shape[2]),dtype=bool)
        for i,u in enumerate(np.where(inRegion)[0]):
            sp[i]=spikes[u,:,:]
            
        changeSp = sp[:,changeFlashes,:]
        preChangeSp = sp[:,np.where(changeFlashes)[0]-1,:]
        hasResp = findResponsiveUnits(preChangeSp,changeSp,baseWin,respWin)
        nUnits = hasResp.sum()
        if nUnits > 0:
            for ind,lbl in zip((changeFlashes,hit,miss,nonChangeFlashes & lick,nonChangeFlashes & ~lick),
                               ('change','hit','miss','non-change lick','non-change no lick')):
                nTrials = ind.sum()
                if nTrials > 0:
                    r = sp[hasResp][:,ind].reshape(nUnits,nTrials,-1,psthBinSize).mean(axis=-1)
                    psth[region][lbl].append(r.mean(axis=1))
                else:
                    psth[region][lbl].append(np.full((nUnits,int(750/psthBinSize)),np.nan))
            psthUnitId[region].append(np.array(units.index[inRegion][hasResp]))
            
for region in regions:
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for lbl,clr in zip(list(psth[region].keys())[1:],'grbk'):
        d = np.concatenate(psth[region][lbl])*1000
        m = np.nanmean(d,axis=0)
        s = np.nanstd(d)/(len(d)**0.5)
        ax.plot(psthTime,m,color=clr,label=lbl)
        ax.fill_between(psthTime,m+s,m-s,color=clr,alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xlim([0,0.75])
    ax.set_xlabel('Time from flash onset (s)')
    ax.set_ylabel('Spikes/s')
    ax.legend(loc='upper right')
    ax.set_title(region + ' (n='+str(len(d))+')')
    plt.tight_layout()


# cluster psth
mbPsth = {lbl: np.concatenate(psth['SC'][lbl]+psth['MRN'][lbl]) for lbl in psth['SC']}
clustData = mbPsth['change'].copy()
clustData -= clustData[:,:int(10/psthBinSize)].mean(axis=1)[:,None]
clustData /= clustData.max(axis=1)[:,None]
clustId,linkageMat = cluster(clustData,nClusters=2,plot=False,colors=None,labels='off',xmax=10.5,nreps=0,title=None)
clustUnitId = np.concatenate(psthUnitId['SC']+psthUnitId['MRN'])

# np.save(os.path.join(outputDir,'sc_mrn_clusterId.npy'),clustId)
# np.save(os.path.join(outputDir,'sc_mrn_clusterUnitId.npy'),clustUnitId)

for clust in (1,2):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for lbl,clr in zip(list(mbPsth.keys())[1:],'grbk'):
        d = mbPsth[lbl][clustId==clust]*1000
        m = np.nanmean(d,axis=0)
        s = np.nanstd(d)/(len(d)**0.5)
        ax.plot(psthTime,m,color=clr,label=lbl)
        ax.fill_between(psthTime,m+s,m-s,color=clr,alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xlim([0,0.75])
    ax.set_xlabel('Time from flash onset (s)')
    ax.set_ylabel('Spikes/s')
    ax.legend(loc='upper right')
    ax.set_title('SC/MRN cluster '+str(clust)+' (n='+str(len(d))+')')
    plt.tight_layout()


# V1 opto
optoTime = np.concatenate(([0],np.array([50,83.33333333,116.66666667])/1000-0.5/60,[0.75]))
optoEffect = 1-np.array([0.07194244604316546,0.14545454545454545,0.5132743362831859,0.7326732673267327,0.7935590421139554])


# summary plot
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(1,1,1)
include = (0,1,2,3,4,5,6,7)
for i,(d,t,clr,lbl) in enumerate(zip((np.concatenate(psth['VISp']['change']),unitChangeDecoding['VISp'],optoEffect,
                                      mbPsth['change'][clustId==1],mbPsth['change'][clustId==2],unitLickDecoding['SC/MRN cluster 1'],
                                      facemapLickDecoding,cumProbLick),
                                     (psthTime,unitDecodingTime,optoTime,
                                      psthTime,psthTime,unitDecodingTime,
                                      facemapDecodingTime,lickLatTime),
                                     ('0.5','r','k','c','m','b','g','k'),
                                     ('V1 spike rate','V1 change decoding','Behavioral effect of V1 silencing',
                                      'SC/MRN spike rate (cluster 1)','SC/MRN spike rate (cluster 2)','SC/MRN cluster 1 lick decoding',
                                      'Face motion lick decoding','Lick probability'))):
    if i not in include:
        ax.plot(np.nan,np.nan,color=clr,label=' '*36)
    elif 'silencing' in lbl:
        m = d-d[-1]
        m /= m.max()
        ax.plot(t,m,'o--',color=clr,label=lbl)
    else:
        m = np.mean(d,axis=0)
        m -= m[0]
        scale = m.max()
        m /= scale
        s = np.std(d,axis=0)/(len(d)**0.5)
        s /= scale
        ax.plot(t,m,color=clr,label=lbl)
        ax.fill_between(t,m+s,m-s,color=clr,alpha=0.25)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlim([0,0.75])
ax.set_ylim([-0.2,1.1])
ax.set_xlabel('Time from flash onset (s)')
ax.set_ylabel('Normalized value')
ax.legend(fontsize=8,bbox_to_anchor=(1,1),loc='upper left')
plt.tight_layout()



# integrator model
filePaths = glob.glob(os.path.join(outputDir,'integratorModel','integratorModel_*.hdf5'))
regions = ('VISall',) #('LGd','VISp','VISl','VISrl','VISal','VISpm','VISam','VISall','SC/MRN cluster 2')
flashLabels = ('change','preChange','catch','nonChange','omitted','prevOmitted','hit','miss','falseAlarm','correctReject')
imageTypeLabels = ('all','familiar','familiarNovel','novel')
imageTypeColors = 'kgm'
modelLabels = ('change','hit','responseTime')
decoderLabels = ('populationAverage','allUnitSpikeCount','allUnitSpikeBins')
t = np.arange(-100,150) 

respMice = {flashes: {imgLbl: [] for imgLbl in imageTypeLabels} for flashes in flashLabels}
respTimeMice = copy.deepcopy(respMice)
respMiceRegions = {region: {flashes: {imgLbl: [] for imgLbl in imageTypeLabels} for flashes in flashLabels} for region in regions}
respTimeMiceRegions = copy.deepcopy(respMiceRegions)
imageNames = copy.deepcopy(respMiceRegions)
integratorInput = copy.deepcopy(respMiceRegions)
integratorValue = {region: {flashes: {imgLbl: {modLbl: [] for modLbl in modelLabels} for imgLbl in imageTypeLabels} for flashes in flashLabels} for region in regions}
respModel = copy.deepcopy(integratorValue)
respTimeModel = copy.deepcopy(integratorValue)
leak = {region: {imgLbl: {modLbl: [] for modLbl in modelLabels} for imgLbl in imageTypeLabels} for region in regions}
threshold = copy.deepcopy(leak)
novelSession = {region: [] for region in regions}
modelAccuracy = {region: {modLbl: [] for modLbl in modelLabels} for region in regions}
trainAccuracy = copy.deepcopy(modelAccuracy)
shuffledAccuracy = copy.deepcopy(modelAccuracy)
decoderAccuracy = {region: {decoder: [] for decoder in decoderLabels} for region in regions}

for i,f in enumerate(filePaths):
    print(i)
    with h5py.File(f,'r') as d:
        if i==0:
            leakRange = d['leakRange'][()]
            thresholdRange = d['thresholdRange'][()]
        novel = d['novel'][()]
        for j,region in enumerate(regions):
            if len(d[region]) > 0:
                novelSession[region].append(np.any(novel))
                for mod in modelLabels:
                    if mod in ('change','hit'):
                        modelAccuracy[region][mod].append(d[region]['integratorAccuracy'][mod][()])
                    trainAccuracy[region][mod].append(d[region]['integratorTrainAccuracy'][mod][()])
                    if mod=='hit':
                        shuffledAccuracy[region][mod].append(d[region]['shuffledAccuracy'][mod][()])
                for decoder in decoderLabels:
                    decoderAccuracy[region][decoder].append(d[region]['decoderAccuracy'][decoder][()])
                intgInput = d[region]['integratorInput'][()]
                intgVal = {mod: d[region]['integratorValue'][mod][()] for mod in modelLabels}
            for k,flashes in enumerate(flashLabels):
                flashInd = d[flashes][()]
                if np.any(novel):
                    inds = (flashInd,(flashInd & ~novel),(flashInd & novel))
                    lbls = ('all',) + imageTypeLabels[-2:]
                else:
                    inds = (flashInd,flashInd)
                    lbls = imageTypeLabels[:2]
                for ind,lbl in zip(inds,lbls):
                    if j==0:
                        respMice[flashes][lbl].append(d['lick'][ind])
                        respTimeMice[flashes][lbl].append(d['lickLatency'][ind])
                    if len(d[region]) > 0:
                        respMiceRegions[region][flashes][lbl].append(d['lick'][ind])
                        respTimeMiceRegions[region][flashes][lbl].append(d['lickLatency'][ind])
                        imageNames[region][flashes][lbl].append(d['imageName'].asstr()[ind])
                        integratorInput[region][flashes][lbl].append(intgInput[ind].mean(axis=0))
                        for mod in modelLabels:
                            integratorValue[region][flashes][lbl][mod].append(intgVal[mod][ind].mean(axis=0))
                            respModel[region][flashes][lbl][mod].append(d[region]['integratorResp'][mod][ind])
                            respTimeModel[region][flashes][lbl][mod].append(d[region]['integratorRespTime'][mod][ind])
                            if k==0:
                                leak[region][lbl][mod].append(d[region]['leak'][mod][()])
                                threshold[region][lbl][mod].append(d[region]['threshold'][mod][()])


# plot mean spike rate
for flashes in ('change','preChange'): #flashLabels:
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for lbl,clr in zip(imageTypeLabels[1:],imageTypeColors):
        d = np.array(integratorInput[region][flashes][lbl])*1000
        m = np.nanmean(d,axis=0)
        n = np.sum(~np.isnan(d[:,0]))
        s = np.nanstd(d,axis=0)/(n**0.5)
        ax.plot(t,m,color=clr,label=lbl+' (n='+str(n)+')')
        ax.fill_between(t,m+s,m-s,color=clr,alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xlim([0,t[-1]])
    # ax.set_ylim([-1,16])
    ax.set_xlabel('Time from flash onset (ms)')
    ax.set_ylabel('Weighted Spikes/s')
    ax.set_title(region+', '+flashes)
    ax.legend(fontsize=8,bbox_to_anchor=(1,1),loc='upper left')
    plt.tight_layout()


# plot mouse response rate and latency
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
xticks = np.arange(len(flashLabels))
for lbl,clr in zip(imageTypeLabels[1:],imageTypeColors):
    mean = []
    sem = []
    for flashes in flashLabels:
        d = [r.sum()/r.size for r in respMice[flashes][lbl]]
        mean.append(np.nanmean(d))
        sem.append(np.nanstd(d)/(np.sum(~np.isnan(d))**0.5))
    ax.plot(xticks,mean,'o',color=clr,ms=10,label=lbl)
    for x,m,s in zip(xticks,mean,sem):
        ax.plot([x,x],[m-s,m+s],color=clr)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xticks(xticks)
ax.set_xticklabels(flashLabels)
ax.set_xlim([xticks[0]-0.25,xticks[-1]+0.25])
ax.set_ylim([0,1])
ax.set_ylabel('Response rate')
ax.set_title('Mice')
ax.legend(loc='upper right')
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for lbl,clr in zip(imageTypeLabels[1:],imageTypeColors):
    mean = []
    sem = []
    for flashes in flashLabels:
        d = [np.nanmean(r)*1000 for r in respTimeMice[flashes][lbl]]
        mean.append(np.nanmean(d))
        sem.append(np.nanstd(d)/(np.sum(~np.isnan(d))**0.5))
    ax.plot(xticks,mean,'o',color=clr,ms=10,label=lbl)
    for x,m,s in zip(xticks,mean,sem):
        ax.plot([x,x],[m-s,m+s],color=clr)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xticks(xticks)
ax.set_xticklabels(flashLabels)
ax.set_xlim([xticks[0]-0.25,xticks[-1]+0.25])
# ax.set_ylim([350,500])
ax.set_ylabel('Response time (ms)')
ax.set_title('Mice')
ax.legend()
plt.tight_layout()


# plot integrator value
for flashes in ('change',):# flashLabels:
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for lbl,clr in zip(imageTypeLabels[1:],imageTypeColors):
        d = np.array(integratorValue[region][flashes][lbl]['change'])
        #d /= np.array(threshold[region][lbl]['change'])[:,-1][:,None]
        m = np.nanmean(d,axis=0)
        n = np.sum(~np.isnan(d[:,0]))
        s = np.nanstd(d,axis=0)/(n**0.5)
        ax.plot(t,m,color=clr,label=lbl+' (n='+str(n)+')')
        ax.fill_between(t,m+s,m-s,color=clr,alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xlim([0,t[-1]])
    ax.set_yticks([0,1])
    # ax.set_ylim([-0.05,1.3])
    ax.set_xlabel('Time from change (ms)')
    ax.set_ylabel('Integrator value relative to threshold')
    ax.set_title(region+', '+flashes)
    ax.legend(fontsize=8,bbox_to_anchor=(1,1),loc='upper left')
    plt.tight_layout()


# plot model response rate and decision time
fig = plt.figure()
xticks = np.arange(len(flashLabels))
ax = fig.add_subplot(1,1,1)
for lbl,clr in zip(imageTypeLabels[1:],imageTypeColors):
    mean = []
    sem = []
    for flashes in flashLabels:
        d = [r.sum()/r.size for r in respModel[region][flashes][lbl]['change']]
        mean.append(np.nanmean(d))
        sem.append(np.nanstd(d)/(np.sum(~np.isnan(d))**0.5))
    ax.plot(xticks,mean,'o',color=clr,ms=8,label=lbl)
    for x,m,s in zip(xticks,mean,sem):
        ax.plot([x,x],[m-s,m+s],color=clr)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xticks(xticks)
ax.set_xticklabels(flashLabels)
ax.set_xlim([xticks[0]-0.25,xticks[-1]+0.25])
ax.set_ylim([0,1])
ax.set_ylabel('Response rate')
ax.set_title('Model')
ax.legend()
plt.tight_layout()

fig = plt.figure()
xticks = np.arange(len(flashLabels))
ax = fig.add_subplot(1,1,1)
for lbl,clr in zip(imageTypeLabels[1:],imageTypeColors):
    mean = []
    sem = []
    for flashes in flashLabels:
        d = [np.nanmean(r) for r in respTimeModel[region][flashes][lbl]['change']]
        mean.append(np.nanmean(d))
        sem.append(np.nanstd(d)/(len(d)**0.5))
    ax.plot(xticks,mean,'o',color=clr,ms=8,label=lbl)
    for x,m,s in zip(xticks,mean,sem):
        ax.plot([x,x],[m-s,m+s],color=clr)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xticks(xticks)
ax.set_xticklabels(flashLabels)
ax.set_xlim([xticks[0]-0.25,xticks[-1]+0.25])
#ax.set_ylim([75,150])
ax.set_ylabel('Decision time (ms)')
ax.set_title('Model')
ax.legend()
plt.tight_layout()


# plot threshold and leak
region = 'VISall'
mod = 'responseTime'
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
extent = [leakRange[0] - 0.5*leakRange[0], leakRange[-1] + 0.5*leakRange[0],
          thresholdRange[0]-0.5*thresholdRange[0], thresholdRange[-1] + 0.5*thresholdRange[0]]
if mod=='responseTime':
    d = np.nanmean(trainAccuracy[region][mod],axis=(0,1)) 
    d[np.isnan(d)] = np.nanmax(d)
else:
    np.array(trainAccuracy[region][mod])[:,-1].mean(axis=0)
im = ax.imshow(d.T,cmap='gray',interpolation='none',extent=extent,aspect='auto',origin='lower')
mean = []
sem = []
for lbl,clr in zip(('familiar','novel'),'gm'):
    lk = np.nanmean(leak[region][lbl][mod],axis=1) if mod=='responseTime' else np.array(leak[region][lbl][mod])[:,-1]
    thresh = np.nanmean(threshold[region][lbl][mod],axis=1) if mod=='responseTime' else np.array(threshold[region][lbl][mod])[:,-1]
    ax.plot(lk,thresh,'o',mec=clr,mfc='none',ms=8)
    mx,my = (lk.mean(),thresh.mean())
    sx,sy = (lk.std()/(len(lk)**0.5),thresh.std()/(len(thresh)**0.5))
    ax.plot(mx,my,'o',color=clr,ms=12)
    ax.plot([mx-sx,mx+sx],[my,my],color=clr)
    ax.plot([mx,mx],[my-sy,my+sy],color=clr)
ax.set_xlabel('leak time constant (ms)')
ax.set_ylabel('threshold (spikes/neuron)')
ax.set_title('average model accuracy')
cb = plt.colorbar(im,ax=ax,fraction=0.05,pad=0.04,shrink=0.5)
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for lbl,clr in zip(('familiar','novel'),'gm'):
    dsort = np.sort(np.array(threshold[region][lbl][mod])[:,-1] )
    cumProb = np.array([np.sum(dsort<=i)/dsort.size for i in dsort])
    ax.plot(dsort,cumProb,color=clr,label=lbl)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_ylim([0,1.01])
ax.set_xlabel('Integrator threshold (spikes/neuron)')
ax.set_ylabel('Cumulative probability')
ax.legend()
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for lbl,clr in zip(('familiar','novel'),'gm'):
    dsort = np.sort(np.array(leak[region][lbl][mod])[:,-1])
    cumProb = np.array([np.sum(dsort<=i)/dsort.size for i in dsort])
    ax.plot(dsort,cumProb,color=clr,label=lbl)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_ylim([0,1.01])
ax.set_xlabel('Integrator leak')
ax.set_ylabel('Cumulative probability')
ax.legend()
plt.tight_layout()


# model accuracy
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
xticks = np.arange(len(regions))
for sessionType,clr in zip(('familiar','novel'),'gm'):
    mean = []
    sem = []
    for region in regions:
        ind = np.array(novelSession[region]) if sessionType=='novel' else ~np.array(novelSession[region])
        d = np.array(modelAccuracy[region]['change'])[ind]
        mean.append(np.mean(d))
        sem.append(np.std(d)/(len(d)**0.5))
    ax.plot(xticks,mean,'o',color=clr,ms=10,label=sessionType)
    for x,m,s in zip(xticks,mean,sem):
        ax.plot([x,x],[m-s,m+s],color=clr)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xticks(xticks)
ax.set_xticklabels(regions)
ax.set_xlim([xticks[0]-0.25,xticks[-1]+0.25])
ax.set_ylim([0.5,1])
ax.set_ylabel('Change detection accuracy')
plt.tight_layout()

for decoder in decoderLabels:
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot([0,1],[0,1],'k--')
    ax.plot(decoderAccuracy[region][decoder],modelAccuracy[region],'o',mec='k',mfc='none',alpha=0.25)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xlim([0.5,1])
    ax.set_ylim([0.5,1])
    ax.set_aspect('equal')
    ax.set_xlabel('Decoder accuracy ('+decoder+')')
    ax.set_ylabel('Integrator accuracy')
    plt.tight_layout()  

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot([0,1],[0,1],'k--')
ax.plot(decoderAccuracy['VISall']['allUnits'],decoderAccuracy['VISall']['populationAverage'],'o',mec='k',mfc='none',alpha=0.25)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlim([0.5,1])
ax.set_ylim([0.5,1])
ax.set_aspect('equal')
ax.set_xlabel('Decoder accuracy (all units)')
ax.set_ylabel('Decoder accuracy (population average)')
plt.tight_layout()                        


# correlation of model and mouse responses and response times
region = 'VISall'
mod = 'responseTime'
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
xticks = np.arange(2)
accuracy = []
chance = []
for mouseChange,mouseCatch,modelChange,modelCatch in zip(respMiceRegions[region]['change']['all'],respMiceRegions[region]['catch']['all'],
                                                         respModel[region]['change']['all'][mod],respModel[region]['catch']['all'][mod]):
    mouse = np.concatenate((mouseChange,mouseCatch))
    model = np.concatenate((modelChange,modelCatch))
    accuracy.append(sklearn.metrics.balanced_accuracy_score(mouse,model))
    chance.append(np.mean([sklearn.metrics.balanced_accuracy_score(mouse,model[np.random.permutation(model.size)]) for _ in range(100)]))
for d,ls,lbl in zip((accuracy,chance),('-','--'),('data','shuffled')):
    dsort = np.sort(d)
    cumProb = np.array([np.sum(dsort<=i)/dsort.size for i in dsort])
    ax.plot(dsort,cumProb,color='k',ls=ls,label=lbl)   
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlim([0.45,1])
ax.set_ylim([0,1.01])
ax.set_xlabel('Similarity of model and mouse (balanced accuracy)')
ax.set_ylabel('Cumulative fraction of sessions')
ax.legend(loc='lower right')
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
xticks = np.arange(2)
accuracy = {lbl: [] for lbl in ('all','familiar','novel')}
chance = copy.deepcopy(accuracy)
for mouse,model,novel in zip(respMiceRegions[region]['change']['all'],respModel[region]['change']['all'][mod],novelSession[region]):
    accuracy['all'].append(sklearn.metrics.balanced_accuracy_score(mouse,model))
    chance['all'].append(np.mean([sklearn.metrics.balanced_accuracy_score(mouse,model[np.random.permutation(model.size)]) for _ in range(100)]))
    lbl = 'novel' if novel else 'familiar'
    accuracy[lbl].append(accuracy['all'][-1])
    chance[lbl].append(chance['all'][-1])
for lbl,clr in zip(('all','familiar','novel'),'kgm'):
    for d,ls,lb in zip((accuracy[lbl],chance[lbl]),('-','--'),('','shuffled')):
        dsort = np.sort(d)
        cumProb = np.array([np.sum(dsort<=i)/dsort.size for i in dsort])
        ax.plot(dsort,cumProb,color=clr,ls=ls,label=lbl+' '+lb)   
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
# ax.set_xlim([0.45,1])
ax.set_ylim([0,1.01])
ax.set_xlabel('Similarity of model and mouse (balanced accuracy)')
ax.set_ylabel('Cumulative fraction of sessions')
ax.legend(loc='lower right')
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
xticks = np.arange(2)
corr = []
chance = []
for mouseChange,mouseCatch,modelChange,modelCatch in zip(respTimeMiceRegions[region]['change']['all'],respTimeMiceRegions[region]['catch']['all'],
                                                         respTimeModel[region]['change']['all'][mod],respTimeModel[region]['catch']['all'][mod]):
    mouse = np.concatenate((mouseChange,mouseCatch))
    model = np.concatenate((modelChange,modelCatch))
    ind = ~np.isnan(mouse) & ~np.isnan(model)
    mouse = mouse[ind]
    model = model[ind]
    corr.append(np.corrcoef(mouse,model)[0,1])
    chance.append(np.mean([np.corrcoef(mouse,model[np.random.permutation(model.size)])[0,1] for _ in range(100)]))
for d,ls,lbl in zip((corr,chance),('-','--'),('data','shuffled')):
    d = np.array(d)
    dsort = np.sort(d[~np.isnan(d)])
    cumProb = np.array([np.sum(dsort<=i)/dsort.size for i in dsort])
    ax.plot(dsort,cumProb,color='k',ls=ls,label=lbl)   
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_ylim([0,1.01])
ax.set_xlabel('Correlation of mouse and model response times')
ax.set_ylabel('Cumulative fraction of sessions')
ax.legend(loc='lower right')
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
xticks = np.arange(2)
corr = {lbl: [] for lbl in ('all','familiar','novel')}
chance = copy.deepcopy(corr)
for mouse,model,novel in zip(respTimeMiceRegions[region]['change']['all'],respTimeModel[region]['change']['all'][mod],novelSession[region]):
    ind = ~np.isnan(mouse) & ~np.isnan(model)
    mouse = mouse[ind]
    model = model[ind]
    corr['all'].append(np.corrcoef(mouse,model)[0,1])
    chance['all'].append(np.mean([np.corrcoef(mouse,model[np.random.permutation(model.size)])[0,1] for _ in range(100)]))
    lbl = 'novel' if novel else 'familiar'
    corr[lbl].append(corr['all'][-1])
    chance[lbl].append(chance['all'][-1])
for lbl,clr in zip(('all','familiar','novel'),'kgm'):
    for d,ls,lb in zip((corr[lbl],chance[lbl]),('-','--'),('','shuffled')):
        d = np.array(d)
        dsort = np.sort(d[~np.isnan(d)])
        cumProb = np.array([np.sum(dsort<=i)/dsort.size for i in dsort])
        ax.plot(dsort,cumProb,color=clr,ls=ls,label=lbl+' '+lb)   
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_ylim([0,1.01])
ax.set_xlabel('Correlation of mouse and model response times')
ax.set_ylabel('Cumulative fraction of sessions')
ax.legend(loc='lower right')
plt.tight_layout()


# image response matrix
region = 'VISall'

images = {'G': ['im012_r','im036_r','im044_r','im047_r','im078_r','im115_r','im083_r','im111_r'],
          'H': ['im005_r','im024_r','im034_r','im087_r','im104_r','im114_r','im083_r','im111_r']}

preImage = [np.concatenate((change,catch)) for change,catch in zip(imageNames[region]['preChange']['all'],imageNames[region]['catch']['all'])]
changeImage = [np.concatenate((change,catch)) for change,catch in zip(imageNames[region]['change']['all'],imageNames[region]['catch']['all'])]
mouseResp = [np.concatenate((change,catch)) for change,catch in zip(respMiceRegions[region]['change']['all'],respMiceRegions[region]['catch']['all'])]
modelResp = [np.concatenate((change,catch)) for change,catch in zip(respModel[region]['change']['all']['change'],respModel[region]['catch']['all']['change'])]
mouseRespTime = [np.concatenate((change,catch)) for change,catch in zip(respTimeMiceRegions[region]['change']['all'],respTimeMiceRegions[region]['catch']['all'])]
modelRespTime = [np.concatenate((change,catch)) for change,catch in zip(respTimeModel[region]['change']['all']['change'],respTimeModel[region]['catch']['all']['change'])]

respMat = {src: {imgSet: {famNov: [] for famNov in ('familiar','novel')} for imgSet in ('G','H')} for src in ('mouse','model')}
respTimeMat = copy.deepcopy(respMat)
for resp,respTime,src in zip((mouseResp,modelResp),(mouseRespTime,modelRespTime),('mouse','model')):
    for novel,preImgs,chImgs,rsp,rspTme in zip(novelSession[region],preImage,changeImage,resp,respTime):
        imgSet = 'G' if np.all(np.in1d(chImgs,images['G'])) else 'H'
        famNov = 'novel' if novel else 'familiar'
        rmat = np.zeros((8,8))
        rcount = rmat.copy()
        rtmat = rmat.copy()
        rtcount = rmat.copy()
        for pre,ch,r,rt in zip(preImgs,chImgs,rsp,rspTme):
            i = images[imgSet].index(pre)
            j = images[imgSet].index(ch)
            rcount[i,j] += 1
            if r:
                rmat[i,j] += 1
            if not np.isnan(rt):
                rtcount[i,j] += 1
                rtmat[i,j] += rt
        rmat /= rcount
        rtmat /= rtcount
        respMat[src][imgSet][famNov].append(rmat)
        respTimeMat[src][imgSet][famNov].append(rtmat)
    
for imgSet in ('G','H'):
    for famNov in ('familiar','novel'):
        fig = plt.figure()
        fig.suptitle('image set '+imgSet+', '+famNov+' n='+str(len(respMat['mouse'][imgSet][famNov]))+' sessions')
        gs = matplotlib.gridspec.GridSpec(2,2)
        for i,src in enumerate(('mouse','model')):
            for j,(r,lbl) in enumerate(zip((respMat,respTimeMat),('response rate','response time (ms)'))):
                ax = fig.add_subplot(gs[i,j])
                m = np.nanmean(r[src][imgSet][famNov],axis=0)
                if src=='mouse' and 'time' in lbl:
                    m *= 1000
                im = ax.imshow(m,cmap='magma',origin='lower')
                cb = plt.colorbar(im,ax=ax,fraction=0.05,pad=0.04,shrink=0.5)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_xlabel('Change image')
                ax.set_ylabel('Pre-change image')
                ax.set_title(src+' '+lbl)
        plt.tight_layout()


# example session
i = 0
for f in filePaths[-1:0:-1]:
    with h5py.File(f,'r') as d:
        novel = d['novel'][()]
        if len(d[region])>0 and np.any(novel) and d['hit'][()].sum()>10:
            thresh = np.mean(d[region]['threshold']['change'][()])
            intg = d[region]['integratorValue']['change'][()]
            flashInd = d['change'][()]
            exampleIntegrator = {}
            for ind,lbl in zip(((flashInd & ~novel),(flashInd & novel)),('familiar','novel')):
                exampleIntegrator[lbl] = intg[ind]/thresh

            fig = plt.figure()
            ax = fig.add_subplot(1,1,1)
            ax.plot([0,t[-1]],[1,1],'--',color='0.5')
            for lbl,clr in zip(('familiar','novel'),'gm'):
                for j,y in enumerate(exampleIntegrator[lbl]):
                    lb = lbl if j==0 else None
                    ax.plot(t,y,color=clr,alpha=0.2,label=lb)
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False)
            ax.set_xlim([0,t[-1]])
            ax.set_xlabel('Time from change (ms)')
            ax.set_ylabel('Integrator value relative to threshold')
            ax.legend(loc='upper left')
            ax.set_title(os.path.basename(f))
            plt.tight_layout()
            
            i += 1
            if i>=20:
                break

