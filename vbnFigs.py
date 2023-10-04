# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 10:06:06 2023

@author: svc_ccg
"""

import glob
import os
import numpy as np
import pandas as pd
import h5py
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
from vbnAnalysisUtils import getBehavData, findNearest


baseDir = r'\\allen\programs\mindscope\workgroups\np-behavior\vbn_data_release\supplemental_tables'

outputDir = r'\\allen\programs\mindscope\workgroups\np-behavior\VBN_video_analysis'

stimTable = pd.read_csv(os.path.join(baseDir,'master_stim_table.csv'))

sessionIds = stimTable['session_id'].unique()


# number of flashes and licks
nFlashes = []
nLicks = []
for sessionId in sessionIds:
    stim = stimTable[(stimTable['session_id']==sessionId) & stimTable['active']].reset_index()
    flashTimes,changeFlashes,nonChangeFlashes,lick,lickTimes,hit = getBehavData(stim)
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
bins = np.arange(0,0.751,1/60)
cumProbLick = []
for sessionId in sessionIds:
    stim = stimTable[(stimTable['session_id']==sessionId) & stimTable['active']].reset_index()
    flashTimes,changeFlashes,nonChangeFlashes,lick,lickTimes,hit = getBehavData(stim)
    if lick[nonChangeFlashes].sum() >= 10:
        lickLatency = (lickTimes - flashTimes)[nonChangeFlashes & lick]
        dsort = np.sort(lickLatency)
        cumProb = np.array([np.sum(dsort<=i)/dsort.size for i in dsort])
        ax.plot(dsort,cumProb,'k',alpha=0.25)
        h = np.histogram(lickLatency,bins)[0]
        cumProbLick.append(np.cumsum(h)/np.sum(h))
m = np.mean(cumProbLick,axis=0)
s = np.std(cumProbLick,axis=0)/(len(cumProbLick)**0.5)
ax.plot(bins[:-1]+0.5/60,m,color='r',lw=2)
ax.fill_between(bins[:-1]+0.5/60,m+s,m-s,color='r',alpha=0.25)
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
        ax.plot(d['decodeWindows'],d['balancedAccuracy'],'k',alpha=0.25)
        facemapLickDecoding.append(d['balancedAccuracy'])
m = np.mean(facemapLickDecoding,axis=0)
s = np.std(facemapLickDecoding,axis=0)/(len(facemapLickDecoding)**0.5)
ax.plot(d['decodeWindows'],m,color='r',lw=2)
ax.fill_between(d['decodeWindows'],m+s,m-s,color='r',alpha=0.25)
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
decodeWindows = d['decodeWindows']
for region,clr,lbl in zip(regions,regionColors,regionLabels):
    if len(unitLickDecoding[region])>0:
        m = np.mean(unitLickDecoding[region],axis=0)
        s = np.std(unitLickDecoding[region],axis=0)/(len(unitLickDecoding[region])**0.5)
        ax.plot(decodeWindows-decodeWindows[0]/2,m,color=clr,label=lbl+' (n='+str(len(unitLickDecoding[region]))+')')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_ylim([0.45,1])
ax.set_xlabel('Time from non-change flash onset (ms)')
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
decodeWindows = d['decodeWindows']
for region,clr,lbl in zip(regions,regionColors,regionLabels):
    if len(unitChangeDecoding[region])>0:
        m = np.mean(unitChangeDecoding[region],axis=0)
        s = np.std(unitChangeDecoding[region],axis=0)/(len(unitChangeDecoding[region])**0.5)
        ax.plot(decodeWindows-decodeWindows[0]/2,m,color=clr,label=lbl+' (n='+str(len(unitChangeDecoding[region]))+')')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_ylim([0.45,1])
ax.set_xlabel('Time from change (ms)')
ax.set_ylabel('Change decoding accuracy')
ax.legend(loc='upper left',fontsize=8)
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
ax.legend(loc='upper right',fontsize=8)
plt.tight_layout()













