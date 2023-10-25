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
from vbnAnalysisUtils import getBehavData, findResponsiveUnits, pca, cluster


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

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
sampleSize = 20
unitLickDecoding = {region: [] for region in regions}
for i,f in enumerate(glob.glob(os.path.join(outputDir,'unitLickDecoding','unbalanced','unitLickDecoding_*.npy'))):
    print(i)
    d = np.load(f,allow_pickle=True).item()
    if d['lick'].sum() >= 10:
        for region in regions:
            a = d[region][sampleSize]['balancedAccuracy']
            if len(a)>0:
                unitLickDecoding[region].append(a)
unitDecodingTime = (d['decodeWindows'] - d['decodeWindows'][0]/2) / 1000
for region,clr,lbl in zip(regions,regionColors,regionLabels):
    if len(unitLickDecoding[region])>0:
        m = np.mean(unitLickDecoding[region],axis=0)
        s = np.std(unitLickDecoding[region],axis=0)/(len(unitLickDecoding[region])**0.5)
        ax.plot(unitDecodingTime,m,color=clr,label=lbl+' (n='+str(len(unitLickDecoding[region]))+')')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_ylim([0.45,1])
ax.set_xlabel('Time from non-change flash onset (s)')
ax.set_ylabel('Lick decoding balanced accuracy')
ax.legend(loc='upper left',fontsize=8)
plt.tight_layout()



# lick decoding accuracy vs unit sample size
unitSampleSize = np.array([1,5,10,15,20,25,30,40,50,60])

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
lickDecodingSampleSize = {region: [] for region in regions}
for i,f in enumerate(glob.glob(os.path.join(outputDir,'unitLickDecoding','unbalanced','unitLickDecoding_*.npy'))):
    print(i)
    d = np.load(f,allow_pickle=True).item()
    if d['lick'].sum() >= 10:
        for region in regions:
            lickDecodingSampleSize[region].append([])
            for sampleSize in unitSampleSize:
                a = d[region][sampleSize]['balancedAccuracy']
                if len(a) > 0:
                    lickDecodingSampleSize[region][-1].append(a[-1])
                else:
                    lickDecodingSampleSize[region][-1].append(np.nan)
for region,clr,lbl in zip(regions,regionColors,regionLabels):
    if len(lickDecodingSampleSize[region])>0:
        m = np.nanmean(lickDecodingSampleSize[region],axis=0)
        n = np.sum(~np.isnan(lickDecodingSampleSize[region]),axis=0)
        s = np.nanstd(lickDecodingSampleSize[region],axis=0)/(n**0.5)
        i = n>2
        ax.plot(unitSampleSize[i],m[i],color=clr,label=lbl)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_ylim([0.45,1])
ax.set_xlabel('Number of units')
ax.set_ylabel('Lick decoding balanced accuracy')
ax.legend(loc='upper right',fontsize=8)
plt.tight_layout()



# unit change decoding
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
sampleSize = 20
unitChangeDecoding = {region: [] for region in regions}
for i,f in enumerate(glob.glob(os.path.join(outputDir,'unitChangeDecoding','unitChangeDecoding_*.npy'))):
    print(i)
    d = np.load(f,allow_pickle=True).item()
    if d['hit'].sum() >= 10:
        for region in regions:
            a = d[region][sampleSize]['accuracy']
            if len(a)>0:
                unitChangeDecoding[region].append(a)
for region,clr,lbl in zip(regions,regionColors,regionLabels):
    if len(unitChangeDecoding[region])>0:
        m = np.mean(unitChangeDecoding[region],axis=0)
        s = np.std(unitChangeDecoding[region],axis=0)/(len(unitChangeDecoding[region])**0.5)
        ax.plot(unitDecodingTime,m,color=clr,label=lbl+' (n='+str(len(unitChangeDecoding[region]))+')')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_ylim([0.45,1])
ax.set_xlabel('Time from change (s)')
ax.set_ylabel('Change decoding accuracy')
ax.legend(loc='lower right',fontsize=8)
plt.tight_layout()



# change decoding accuracy vs unit sample size
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
changeDecodingSampleSize = {region: [] for region in regions}
for i,f in enumerate(glob.glob(os.path.join(outputDir,'unitChangeDecoding','unitChangeDecoding_*.npy'))):
    print(i)
    d = np.load(f,allow_pickle=True).item()
    if d['hit'].sum() >= 10:
        for region in regions:
            changeDecodingSampleSize[region].append([])
            for sampleSize in unitSampleSize:
                a = d[region][sampleSize]['accuracy']
                if len(a) > 0:
                    changeDecodingSampleSize[region][-1].append(a[-1])
                else:
                    changeDecodingSampleSize[region][-1].append(np.nan)
for region,clr,lbl in zip(regions,regionColors,regionLabels):
    if len(changeDecodingSampleSize[region])>0:
        m = np.nanmean(changeDecodingSampleSize[region],axis=0)
        n = np.sum(~np.isnan(changeDecodingSampleSize[region]),axis=0)
        s = np.nanstd(changeDecodingSampleSize[region],axis=0)/(n**0.5)
        i = n>2
        ax.plot(unitSampleSize[i],m[i],color=clr,label=lbl)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_ylim([0.45,1])
ax.set_xlabel('Number of units')
ax.set_ylabel('Change decoding accuracy')
ax.legend(loc='lower right',fontsize=8)
plt.tight_layout()



# change decoding correlation with behavior
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
sampleSize = 20
changeDecodingCorr = {region: [] for region in regions}
for i,f in enumerate(glob.glob(os.path.join(outputDir,'unitChangeDecoding','unitChangeDecoding_*.npy'))):
    print(i)
    d = np.load(f,allow_pickle=True).item()
    hit = d['hit']
    decodeWindows = d['decodeWindows']
    if hit.sum() >= 10 and hit.sum() < hit.size:
        for region in regions:
            conf = d[region][sampleSize]['confidence']
            if len(conf)>0:
                changeDecodingCorr[region].append([np.corrcoef(hit,c[:hit.size])[0,1] for c in conf])
for region,clr,lbl in zip(regions,regionColors,regionLabels):
    if len(changeDecodingCorr[region])>0:
        m = np.mean(changeDecodingCorr[region],axis=0)
        s = np.std(changeDecodingCorr[region],axis=0)/(len(changeDecodingCorr[region])**0.5)
        ax.plot(unitDecodingTime,m,color=clr,label=lbl+' (n='+str(len(changeDecodingCorr[region]))+')')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_ylim([-0.1,1])
ax.set_xlabel('Time from change (s)')
ax.set_ylabel('Change decoder correlation with behavior')
ax.legend(loc='upper left',fontsize=8)
plt.tight_layout()


# psth
regions = ('VISp',('SCig','SCiw'),'MRN')

baseWin = slice(680,750)
respWin = slice(30,100)

psthBinSize = 5
psthTime = np.arange(0,750,psthBinSize)/1000
psth = {region: [] for region in regions}
psthUnitId = copy.deepcopy(psth)
for sessionInd,sessionId in enumerate(sessionIds):
    print(sessionInd)
    units = unitTable.set_index('unit_id').loc[unitData[str(sessionId)]['unitIds'][:]]
    spikes = unitData[str(sessionId)]['spikes']

    stim = stimTable[(stimTable['session_id']==sessionId) & stimTable['active']].reset_index()
    flashTimes,changeFlashes,catchFlashes,nonChangeFlashes,omittedFlashes,prevOmittedFlashes,novelFlashes,lick,lickTimes = getBehavData(stim)
    nChange = changeFlashes.sum()
    
    for region in regions:
        inRegion = np.in1d(units['structure_acronym'],region)
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
            r = changeSp[hasResp].reshape(nUnits,nChange,-1,psthBinSize).mean(axis=-1)
            psth[region].append(r.mean(axis=1))
            psthUnitId[region].append(np.array(units.index[inRegion][hasResp]))
        

# cluster psth    
mbPsth = np.concatenate(psth[('SCig','SCiw')]+psth['MRN'])
clustData = mbPsth.copy()
clustData -= clustData[:,:int(10/psthBinSize)].mean(axis=1)[:,None]
clustData /= clustData.max(axis=1)[:,None]
clustId,linkageMat = cluster(clustData,nClusters=2,plot=False,colors=None,labels='off',xmax=10.5,nreps=0,title=lbl)
clustUnitId = np.concatenate(psthUnitId[('SCig','SCiw')]+psthUnitId['MRN'])


# V1 opto
optoTime = np.concatenate(([0],np.array([50,83.33333333,116.66666667])/1000-0.5/60,[0.75]))
optoEffect = 1-np.array([0.07194244604316546,0.14545454545454545,0.5132743362831859,0.7326732673267327,0.7935590421139554])


# summary plot
fig = plt.figure(figsize=(10,6))
ax = fig.add_subplot(1,1,1)
include = (0,1,2,3,4,5,6,7,8,9)
for i,(d,t,clr,lbl) in enumerate(zip((np.concatenate(psth['VISp']),unitChangeDecoding['VISp'],optoEffect,
                                      mbPsth[clustId==1],mbPsth[clustId==2],
                                      unitChangeDecoding[('SCig','SCiw')]+unitChangeDecoding['MRN'],
                                      changeDecodingCorr[('SCig','SCiw')]+changeDecodingCorr['MRN'],
                                      unitLickDecoding[('SCig','SCiw')]+unitLickDecoding['MRN'],
                                      facemapLickDecoding,cumProbLick),
                                     (psthTime,unitDecodingTime,optoTime,psthTime,psthTime,unitDecodingTime,unitDecodingTime,
                                      unitDecodingTime,facemapDecodingTime,lickLatTime),
                                     ('0.5','r','k','y',[1,0.5,0],'m','c','b','g','k'),
                                     ('V1 spike rate','V1 change decoding','Behavioral effect of V1 silencing',
                                      'SC/MRN spike rate (cluster 1)','SC/MRN spike rate (cluster 2)',
                                      'SC/MRN change decoding','SC/MRN change decoder correlation with behavior',
                                      'SC/MRN lick decoding (non-change flashes)','Face motion lick decoding (non-change flashes)',
                                      'Lick probability'))):
    if i not in include:
        ax.plot(np.nan,np.nan,color=clr,label='                                               ')
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
ax.legend(loc='lower right',fontsize=8)
ax.legend(fontsize=8,bbox_to_anchor=(1,1),loc='upper left')
plt.tight_layout()


# summary plot, simpler version
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
mbReg = ('SCig','SCiw')
mbLbl = 'SC'
# mbReg = 'MRN'
# mbLbl = 'MRN'
for d,t,clr,lbl in zip((cumProbLick,facemapLickDecoding,unitLickDecoding[mbReg],unitChangeDecoding['VISp']),
                       (lickLatTime,facemapDecodingTime)+(unitDecodingTime,)*2,
                       ('k','r','b','g'),
                       ('lick prob','lick decoding (face motion, non-change flashes)',
                        'lick decoding ('+mbLbl+' units, non-change flashes)','change decoding (V1 units)')):
    m = np.mean(d,axis=0)
    m -= m[0]
    scale = 1/m[-1]
    m *= scale
    s = np.std(d,axis=0)/(len(cumProbLick)**0.5)
    s *= scale
    ax.plot(t,m,color=clr,label=lbl)
    ax.fill_between(t,m+s,m-s,color=clr,alpha=0.25)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlim([0,0.75])
ax.set_ylim([-0.05,1.05])
ax.set_xlabel('Time from flash onset (s)')
ax.set_ylabel('Normalized value')
ax.legend(loc='lower right',fontsize=8)
# ax.legend(loc='upper left',fontsize=8)
plt.tight_layout()



# integrator model
filePaths = glob.glob(os.path.join(outputDir,'integratorModel','integratorModel_*.hdf5'))
regions = ('VISall','VISp') #('LGd','VISp','VISl','VISrl','VISal','VISpm','VISam','VISall','SC','MRN')
flashLabels = ('change','preChange','catch','nonChange','omitted','prevOmitted','hit','miss','falseAlarm','correctReject')
imageTypeLabels = ('all','familiar','familiarNovel','novel')
imageTypeColors = 'kgm'
t = np.arange(150) 

respMice = {flashes: {lbl: [] for lbl in imageTypeLabels} for flashes in flashLabels}
respTimeMice = copy.deepcopy(respMice)
respMiceRegions = {region: {flashes: {lbl: [] for lbl in imageTypeLabels} for flashes in flashLabels} for region in regions}
respTimeMiceRegions = copy.deepcopy(respMiceRegions)
spikeRate = copy.deepcopy(respMiceRegions)
integrator = copy.deepcopy(respMiceRegions)
respModel = copy.deepcopy(respMiceRegions)
respTimeModel = copy.deepcopy(respMiceRegions)
imageNames = copy.deepcopy(respMiceRegions)
leak = {region: {lbl: [] for lbl in imageTypeLabels} for region in regions}
threshold = copy.deepcopy(leak)
maxSpikeRate = copy.deepcopy(leak)
novelSession = {region: [] for region in regions}
modelAccuracy = copy.deepcopy(novelSession)
trainAccuracy = copy.deepcopy(novelSession)
decoderAccuracy = {region: {decoder: [] for decoder in ('allUnits','populationAverage')} for region in regions}

for i,f in enumerate(filePaths):
    print(i)
    with h5py.File(f,'r') as d:
        if i==0:
            leakRange = d['leakRange'][()]
            thresholdRange = d['thresholdRange'][()]
        if d['hit'][()].sum() < 10:
            continue
        novel = d['novel'][()]
        for j,region in enumerate(regions):
            if len(d[region]) > 0:
                novelSession[region].append(np.any(novel))
                modelAccuracy[region].append(d[region]['modelAccuracy'][()])
                trainAccuracy[region].append(d[region]['trainAccuracy'][()])
                for decoder in ('allUnits','populationAverage'):
                    decoderAccuracy[region][decoder].append(d[region]['decoderAccuracy'][decoder][()])
                spRate = d[region]['spikeRate'][()]
                intg = d[region]['integrator'][()]
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
                        spikeRate[region][flashes][lbl].append(spRate[ind].mean(axis=0))
                        integrator[region][flashes][lbl].append(intg[ind].mean(axis=0))
                        respModel[region][flashes][lbl].append(d[region]['modelResp'][ind])
                        respTimeModel[region][flashes][lbl].append(d[region]['modelRespTime'][ind])
                        imageNames[region][flashes][lbl].append(d['imageName'].asstr()[ind])
                        if k==0:
                            leak[region][lbl].append(d[region]['leak'][()])
                            threshold[region][lbl].append(d[region]['threshold'][()])
                            maxSpikeRate[region][lbl].append(d[region]['maxSpikeRate'][()])


# plot mean spike rate
for flashes in flashLabels:
    fig = plt.figure(figsize=(8,8))
    fig.suptitle(flashes)
    for i,region in enumerate(regions):
        ax = fig.add_subplot(len(regions),1,i+1)
        for lbl,clr in zip(imageTypeLabels,imageTypeColors):
            d = np.array(spikeRate[region][flashes][lbl])
            m = np.nanmean(d,axis=0)
            n = np.sum(~np.isnan(d[:,0]))
            s = np.nanstd(d,axis=0)/(n**0.5)
            ax.plot(t[1:],m,color=clr,label=lbl+' (n='+str(n)+')')
            ax.fill_between(t[1:],m+s,m-s,color=clr,alpha=0.25)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_xlim([0,t[-1]])
        if i==len(regions)-1:
            ax.set_xlabel('Time from change (ms)')
        else:
            ax.set_xticklabels([])
        ax.set_ylabel('Spikes/s')
        ax.set_title(region)
        ax.legend(fontsize=8,bbox_to_anchor=(1,1),loc='upper left')
    plt.tight_layout()


# plot mouse response rate and latency
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
xticks = np.arange(len(flashLabels))
for lbl,clr in zip(imageTypeLabels,imageTypeColors):
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
for lbl,clr in zip(imageTypeLabels,imageTypeColors):
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
plt.tight_layout()


# plot integrator value
for flashes in flashLabels:
    fig = plt.figure(figsize=(8,8))
    fig.suptitle(flashes)
    for i,region in enumerate(regions):
        ax = fig.add_subplot(len(regions),1,i+1)
        for lbl,clr in zip(imageTypeLabels,imageTypeColors):
            d = np.array(integrator[region][flashes][lbl])
            d /= np.mean(threshold[region][lbl],axis=1)[:,None]
            m = np.nanmean(d,axis=0)
            n = np.sum(~np.isnan(d[:,0]))
            s = np.nanstd(d,axis=0)/(n**0.5)
            ax.plot(t,m,color=clr,label=lbl+' (n='+str(n)+')')
            ax.fill_between(t,m+s,m-s,color=clr,alpha=0.25)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_xlim([0,t[-1]])
        if i==len(regions)-1:
            ax.set_xlabel('Time from change (ms)')
        else:
            ax.set_xticklabels([])
        if i==len(regions)//2-1:
            ax.set_ylabel('Integrator value relative to threshold')
        ax.set_title(region)
        ax.legend(fontsize=8,bbox_to_anchor=(1,1),loc='upper left')
    plt.tight_layout()


# plot model response rate and decision time
fig = plt.figure(figsize=(8,10))
xticks = np.arange(len(flashLabels))
for i,region in enumerate(regions):
    ax = fig.add_subplot(len(regions),1,i+1)
    for lbl,clr in zip(imageTypeLabels,imageTypeColors):
        mean = []
        sem = []
        for flashes in flashLabels:
            d = [r.sum()/r.size for r in respModel[region][flashes][lbl]]
            mean.append(np.nanmean(d))
            sem.append(np.nanstd(d)/(np.sum(~np.isnan(d))**0.5))
        ax.plot(xticks,mean,'o',color=clr,ms=8,label=lbl)
        for x,m,s in zip(xticks,mean,sem):
            ax.plot([x,x],[m-s,m+s],color=clr)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xticks(xticks)
    if i==len(regions)-1:
        ax.set_xticklabels(flashLabels)
    else:
        ax.set_xticklabels([])
    ax.set_xlim([xticks[0]-0.25,xticks[-1]+0.25])
    ax.set_ylim([0,1])
    if i==len(regions)//2-1:
        ax.set_ylabel('Response rate')
    ax.set_title(region)
    if i==0:
        ax.legend(bbox_to_anchor=(1,1),loc='upper left')
plt.tight_layout()

fig = plt.figure(figsize=(8,10))
xticks = np.arange(len(flashLabels))
for i,region in enumerate(regions):
    ax = fig.add_subplot(len(regions),1,i+1)
    for lbl,clr in zip(imageTypeLabels,imageTypeColors):
        mean = []
        sem = []
        for flashes in flashLabels:
            d = [np.nanmean(r) for r in respTimeModel[region][flashes][lbl]]
            mean.append(np.nanmean(d))
            sem.append(np.nanstd(d)/(len(d)**0.5))
        ax.plot(xticks,mean,'o',color=clr,ms=8,label=lbl)
        for x,m,s in zip(xticks,mean,sem):
            ax.plot([x,x],[m-s,m+s],color=clr)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xticks(xticks)
    if i==len(regions)-1:
        ax.set_xticklabels(flashLabels)
    else:
        ax.set_xticklabels([])
    ax.set_xlim([xticks[0]-0.25,xticks[-1]+0.25])
    ax.set_ylim([75,150])
    if i==len(regions)//2-1:
        ax.set_ylabel('Decision time (ms)')
    ax.set_title(region)
    if i==0:
        ax.legend(bbox_to_anchor=(1,1),loc='upper left')
plt.tight_layout()


# plot threshold and leak
region = 'VISall'
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
extent = [leakRange[0] - 0.5*leakRange[0], leakRange[-1] + 0.5*leakRange[0],
          thresholdRange[0]-0.5*thresholdRange[0], thresholdRange[-1] + 0.5*thresholdRange[0]]
im = ax.imshow(np.concatenate(trainAccuracy[region]).mean(axis=0).T,cmap='gray',interpolation='none',extent=extent,aspect='auto',origin='lower')
mean = []
sem = []
for lbl,clr in zip(('familiar','novel'),'gm'):
    lk = np.median(leak[region][lbl],axis=1)
    thresh = np.median(threshold[region][lbl],axis=1)
    ax.plot(lk,thresh,'o',mec=clr,mfc='none',ms=8)
    mx,my = (lk.mean(),thresh.mean())
    sx,sy = (lk.std()/(len(lk)**0.5),thresh.std()/(len(thresh)**0.5))
    ax.plot(mx,my,'o',color=clr,ms=12)
    ax.plot([mx-sx,mx+sx],[my,my],color=clr)
    ax.plot([mx,mx],[my-sy,my+sy],color=clr)
ax.set_xlabel('leak')
ax.set_ylabel('threshold')
ax.set_title('model accuracy')
cb = plt.colorbar(im,ax=ax,fraction=0.05,pad=0.04,shrink=0.5)
plt.tight_layout()

fig = plt.figure(figsize=(6,8))
for i,region in enumerate(regions):
    ax = fig.add_subplot(len(regions),1,i+1)
    for lbl,clr in zip(('familiar','novel'),'gm'):
        thresh = np.mean(threshold[region][lbl],axis=1) 
        thresh *= np.array(maxSpikeRate[region][lbl])/1000
        dsort = np.sort(thresh)
        cumProb = np.array([np.sum(dsort<=i)/dsort.size for i in dsort])
        ax.plot(dsort,cumProb,color=clr,label=lbl)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_ylim([0,1.01])
    if i==len(regions)-1:
        ax.set_xlabel('Integrator threshold (spikes/neuron)')
    else:
        ax.set_xticklabels([])
    if i==len(regions)//2-1:
        ax.set_ylabel('Cumulative probability')
    ax.set_title(region)
    if i==0:
        ax.legend(bbox_to_anchor=(1,1),loc='upper left')
plt.tight_layout()

fig = plt.figure(figsize=(6,8))
for i,region in enumerate(regions):
    ax = fig.add_subplot(len(regions),1,i+1)
    for lbl,clr in zip(('familiar','novel'),'gm'):
        dsort = np.sort(np.mean(leak[region][lbl],axis=1))
        cumProb = np.array([np.sum(dsort<=i)/dsort.size for i in dsort])
        ax.plot(dsort,cumProb,color=clr,label=lbl)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_ylim([0,1.01])
    if i==len(regions)-1:
        ax.set_xlabel('Integrator leak')
    else:
        ax.set_xticklabels([])
    if i==len(regions)//2-1:
        ax.set_ylabel('Cumulative probability')
    ax.set_title(region)
    if i==0:
        ax.legend(bbox_to_anchor=(1,1),loc='upper left')
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
        d = np.array(modelAccuracy[region])[ind]
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

for decoder in ('allUnits','populationAverage'):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot([0,1],[0,1],'k--')
    ax.plot(decoderAccuracy['VISall'][decoder],modelAccuracy['VISall'],'o',mec='k',mfc='none',alpha=0.25)
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
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
xticks = np.arange(2)
accuracy = []
chance = []
for mouseChange,mouseCatch,modelChange,modelCatch in zip(respMiceRegions[region]['change']['all'],respMiceRegions[region]['catch']['all'],
                                                         respModel[region]['change']['all'],respModel[region]['catch']['all']):
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
for mouse,model,novel in zip(respMiceRegions[region]['change']['all'],respModel[region]['change']['all'],novelSession[region]):
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
ax.set_xlim([0.45,1])
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
                                                         respTimeModel[region]['change']['all'],respTimeModel[region]['catch']['all']):
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
for mouse,model,novel in zip(respTimeMiceRegions[region]['change']['all'],respTimeModel[region]['change']['all'],novelSession[region]):
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
mouseResp,modelResp = [[np.concatenate((change,catch)) for change,catch in zip(r[region]['change']['all'],r[region]['catch']['all'])] for r in (respMiceRegions,respModel)]
mouseRespTime,modelRespTime = [[np.concatenate((change,catch)) for change,catch in zip(r[region]['change']['all'],r[region]['catch']['all'])] for r in (respTimeMiceRegions,respTimeModel)]

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
f = filePaths[0]
with h5py.File(f,'r') as d:
    pass




