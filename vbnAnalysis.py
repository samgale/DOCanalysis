# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 16:22:30 2022

@author: svc_ccg
"""

import copy
import os
import numpy as np
import scipy.stats
import pandas as pd
import h5py
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42



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


sessionIds = stimTable['session_id'][stimTable['experience_level']=='Familiar'].unique()

regions = ('LGd','VISp','VISl','VISrl','VISal','VISpm','VISam','LP','MRN')
layers = ('1','2/3','4','5','6a','6b')

binSize = 0.001
baseWin = slice(690,750)
respWin = slice(40,100)


def findResponsiveUnits(basePsth,respPsth,baseWin,respWin):
    base = basePsth[:,:,baseWin].mean(axis=1)
    resp = respPsth[:,:,respWin].mean(axis=1)
    peak = np.max(resp-base.mean(axis=1)[:,None],axis=1)
    hasResp = peak > 5 * base.std()
    
    base = basePsth[:,:,baseWin].mean(axis=2)
    resp = respPsth[:,:,respWin].mean(axis=2)
    pval = np.array([1 if np.sum(r-b)==0 else scipy.stats.wilcoxon(b,r)[1] for b,r in zip(base,resp)])
    
    return hasResp & (pval<0.05)

    
#
unitCount = np.zeros((len(sessionIds),len(regions)),dtype=int)
for i,sid in enumerate(sessionIds):
    units = unitTable.set_index('unit_id').loc[unitData[str(sid)]['unitIds'][:]]
    for j,reg in enumerate(regions):
        unitCount[i,j] = np.sum(units['structure_acronym']==reg)


#
changeSpikes = {region: {layer: [] for layer in layers} for region in regions}
preChangeSpikes = copy.deepcopy(changeSpikes)    
adaptSpikes = copy.deepcopy(changeSpikes)
for si,sid in enumerate(sessionIds):
    units = unitTable.set_index('unit_id').loc[unitData[str(sid)]['unitIds'][:]]
    spikes = unitData[str(sid)]['spikes']
    
    stim = stimTable[(stimTable['session_id']==sid) & stimTable['active']]
    autoRewarded = np.array(stim['auto_rewarded']).astype(bool)
    changeFlash = np.where(stim['is_change'] & ~autoRewarded)[0]
    notOmitted = [flash for flash in changeFlash if not any(stim[flash-1:flash+10]['omitted']) and flash+10<spikes.shape[1]]
    
    for region in regions:
        inRegion = np.array(units['structure_acronym']==region)
        if not any(inRegion):
            continue
        for layer in layers:
            print('session '+str(si+1)+', '+region+', '+layer)
            inLayer = inRegion & np.array(units['cortical_layer']==layer) if 'VIS' in region else inRegion
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
            changeSpikes[region][layer].append(changeSp[hasResp].mean(axis=1))
            preChangeSpikes[region][layer].append(preChangeSp[hasResp].mean(axis=1))
            adaptSp = np.zeros((hasResp.sum(),len(notOmitted),11*750),dtype=bool)
            for i,flash in enumerate(notOmitted):
                adaptSp[:,i,:] = s[hasResp,flash-1:flash+10,:].reshape((hasResp.sum(),-1))
            adaptSpikes[region][layer].append(adaptSp.mean(axis=1))
            if not 'VIS' in region:
                break

t = np.arange(11*750)/1000 - 750

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for region,clr in zip(regions,plt.cm.magma(np.linspace(0,1,len(regions)))):
    if 'VIS' in region:
        d = np.concatenate([np.concatenate(adaptSpikes[region][layer]) for layer in layers if len(adaptSpikes[region][layer])>0])
    else:
        d = np.concatenate(adaptSpikes[region]['1'])
    d -= d[:,500:750].mean(axis=1)[:,None]
    d /= binSize
    ax.plot(t,d.mean(axis=0),color=clr,label=region)            
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlim([-0.25,7.5])
ax.set_xlabel('Time (s)')
ax.set_ylabel('Spikes/s')
ax.legend()
plt.tight_layout()


















