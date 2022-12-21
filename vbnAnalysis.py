# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 16:22:30 2022

@author: svc_ccg
"""

import os
import numpy as np
import pandas as pd
import h5py
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
    




#
baseDir = r"C:\Users\svc_ccg\Desktop\Analysis\vbn"

stimTable = pd.read_csv(os.path.join(baseDir,'master_stim_table.csv'))

unitTable = pd.read_csv(os.path.join(baseDir,'units_with_cortical_layers.csv'))

unitData = h5py.File(os.path.join(baseDir,'vbnAllUnitSpikeTensor.hdf5'),mode='r')


def findResponsiveUnits(base,resp,baseWin,respWin):
    baseMean = base[:,:,baseWin].mean(axis=1)
    respMean = resp[:,:,respWin].mean(axis=1)
    peak = (m-b.mean())[analysisWindow].max()
    
    base = base[:,:,baseWin].mean(axis=2)
    resp = resp[:,:,respWin].mean(axis=2)
    pval = np.array([1 if np.sum(r-b)==0 else scipy.stats.wilcoxon(b,r)[1] for b,r in zip(base,resp)])
    


sessionIds = stimTable['session_id'][stimTable['experience_level']=='Familiar'].unique()

regions = ('LGd','VISp','VISl','VISrl','VISal','VISpm','VISam','LP','MRN')
layers = ('all','1','2/3','4','5',('6a','6b'))

baselineWindow = slice(690,750)
responseWindow = slice(40,100)

unitCount = np.zeros((len(sessionIds),len(regions)),dtype=int)
for i,sid in enumerate(sessionIds):
    units = unitTable.set_index('unit_id').loc[unitData[str(sid)]['unitIds'][:]]
    for j,reg in enumerate(regions):
        unitCount[i,j] = np.sum(units['structure_acronym']==reg)

preChangePsth = []
changePsth = []    
adaptPsth = []
for sid in sessionIds:
    units = unitTable.set_index('unit_id').loc[unitData[str(sid)]['unitIds'][:]]
    spikes = unitData[str(sid)]['spikes']
    
    stim = stimTable[(stimTable['session_id']==sid) & stimTable['active']]
    autoRewarded = np.array(stim['auto_rewarded']).astype(bool)
    isChange = stim['is_change'] & ~autoRewarded
    changeFlash = np.where(isChange)[0]
    preChangeFlash = changeFlash - 1
    hasOmitted = np.array([any(stim[(stim['behavior_trial_id']==trial)]['omitted']) for trial in stim[isChange]['behavior_trial_id']])
    
    for region in regions:
        inRegion = np.array(units['structure_acronym']==region)
        for layer in layers:
            inLayer = np.array(units['cortical_layer']==layer) & inRegion if 'VIS' in region and layer!='all' else inRegion
            s = spikes[inLayer,:,:]
            for flash in changeFlash:
                preChangePsth = s[:,flash-1,:]
                changePsth = s[:,flash,:]
                if not any(stim[flash-1:flash+10]['ommitted'] and flash+10<s.shape[1]:
                    adapthPsth = s[:,flash-1:flash+10,:].reshape((s.shape[0],-1)).transpose((1,0,2))

            
    



















