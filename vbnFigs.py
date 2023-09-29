# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 10:06:06 2023

@author: svc_ccg
"""

import glob, os, time, warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
import h5py


baseDir = r'\\allen\programs\mindscope\workgroups\np-behavior\VBN_video_analysis'


decodeWindowEnd = 0.75
frameInterval = 1/60
decodeWindows = np.arange(0,decodeWindowEnd+frameInterval/2,frameInterval)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for f in glob.glob(os.path.join(baseDir,'facemapLickDecoding','facemapLickDecoding_*.npy')):
    d = np.load(f,allow_pickle=True).item()
    ax.plot(decodeWindows,d['balancedAccuracy'],'k',alpha=0.25)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_ylim([0.5,1])
ax.set_xlabel('Time from non-change flash onset (ms)')
ax.set_ylabel('Lick decoding balanced accuracy')
plt.tight_layout()


#
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


decodeWindowSize = 10
decodeWindowEnd = 750
decodeWindows = np.arange(decodeWindowSize,decodeWindowEnd+decodeWindowSize,decodeWindowSize)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
d = {region: [] for region in regions}
for f in glob.glob(os.path.join(baseDir,'unitLickDecoding','unitLickDecoding_*.npy')):
    data = np.load(f,allow_pickle=True).item()
    for region in regions:
        if len(data[region]['balancedAccuracy'])>0:
            d[region].append(data[region]['balancedAccuracy'])
for region,clr,lbl in zip(regions,regionColors,regionLabels):
    if len(d[region])>0:
        m = np.mean(d[region],axis=0)
        s = np.std(d[region],axis=0)/(len(d[region])**0.5)
        ax.plot(decodeWindows-decodeWindowSize/2,m,color=clr,label=lbl+' (n='+str(len(d[region]))+')')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_ylim([0.45,1])
ax.set_xlabel('Time from non-change flash onset (ms)')
ax.set_ylabel('Lick decoding balanced accuracy')
ax.legend(loc='upper left',fontsize=8)
plt.tight_layout()

















