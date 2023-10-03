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
from vbnAnalysisUtils import getFlashTimes


baseDir = r'\\allen\programs\mindscope\workgroups\np-behavior\vbn_data_release\supplemental_tables'

outputDir = r'\\allen\programs\mindscope\workgroups\np-behavior\VBN_video_analysis'

stimTable = pd.read_csv(os.path.join(baseDir,'master_stim_table.csv'))

sessionIds = stimTable['session_id'].unique()



#
nFlashes = []
nLicks = []
for sessionId in sessionIds:
    stim = stimTable[(stimTable['session_id']==sessionId) & stimTable['active']].reset_index()
    flashTimes,changeFlashes,nonChangeFlashes,lick = getFlashTimes(stim)
    nFlashes.append(nonChangeFlashes.sum())
    nLicks.append(lick[nonChangeFlashes].sum())
nFlashes = np.array(nFlashes)
nLicks = np.array(nLicks)
lickProb = nLicks / nFlashes

for d in (nFlashes,nLicks,lickProb):
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.hist(d)
    # dsort = np.sort(d)
    # cumProb = np.array([np.sum(d<=i)/d.size for i in dsort])
    # ax.plot(dsort,cumProb,'k')
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    plt.tight_layout()
    


#
for f in glob.glob(os.path.join(outputDir,'facemapData','*.hdf5')):
    sessionId = os.path.basename(f)[:10]
    fig = plt.figure(figsize=(3,6))
    with h5py.File(f) as d:
        x = d['xrange_bin'][()]
        y = d['yrange_bin'][()]
        
        ax = fig.add_subplot(3,1,1)
        ax.imshow(d['avgframe'][()][y,:][:,x],cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(sessionId + '\n' + 'average frame')
        
        ax = fig.add_subplot(3,1,2)
        ax.imshow(d['avgmotion'][()][y,:][:,x],cmap='gray')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('average motion')
        
        ax = fig.add_subplot(3,1,3)
        ax.imshow(np.average(d['motMask'][()],weights=d['motSv'][()],axis=2),cmap='bwr')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title('weighted average\nmask')
    plt.tight_layout()
    plt.savefig(os.path.join(outputDir,'facemapRois',sessionId+'_roi.png'))
    plt.close()
    
    
#
s = []
for f in glob.glob(os.path.join(outputDir,'facemapLickDecoding','facemapLickDecoding_*.npy')):
    d = np.load(f,allow_pickle=True).item()
    a = d['balancedAccuracy']
    if a[0] > 0.6 or a[-1] < 0.8:
        s.append((os.path.basename(f)[-14:-5],a[0],a[-1]))
        



#
fig = plt.figure(figsize=(6,6))
gs = matplotlib.gridspec.GridSpec(11,10)
i = 0
j = 0
for f in glob.glob(os.path.join(outputDir,'facemapData','*.hdf5')):
    ax = fig.add_subplot(gs[i,j])
    with h5py.File(f) as d:
        x = d['xrange_bin'][()]
        y = d['yrange_bin'][()]
        ax.imshow(d['avgframe'][()][y,:][:,x],cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    if j == 9:
        i += 1
        j = 0
    else:
        j += 1
plt.tight_layout()






#
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
a = []
for f in glob.glob(os.path.join(outputDir,'facemapLickDecoding','facemapLickDecoding_*.npy')):
    d = np.load(f,allow_pickle=True).item()
    if d['lick'].sum() >= 10:
        ax.plot(d['decodeWindows'],d['balancedAccuracy'],'k',alpha=0.25)
        a.append(d['balancedAccuracy'])
m = np.mean(a,axis=0)
s = np.std(a,axis=0)/(len(a)**0.5)
ax.plot(d['decodeWindows'],m,color='r',lw=2)
ax.fill_between(d['decodeWindows'],m+s,m-s,color='r',alpha=0.25)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_ylim([0.45,1])
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
for f in glob.glob(os.path.join(outputDir,'unitLickDecoding','unitLickDecoding_*.npy')):
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

















