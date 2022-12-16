# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 16:22:30 2022

@author: svc_ccg
"""

import numpy as np
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
    
    spikes = np.zeros((len(goodUnits),len(flashTimes),int(windowDur/binSize)),dtype=bool)
    i = 0
    for unitId,unitData in goodUnits.iterrows(): 
        spikes[i] = getSpikeBins(spikeTimes[unitId],flashTimes,windowDur,binSize)
        i += 1
    
    h5Group = h5File.create_group(str(sessionId))
    h5Group.create_dataset('unitIds',data=goodUnits.index,compression='gzip',compression_opts=4)
    h5Group.create_dataset('spikes',data=spikes,chunks=(1,spikes.shape[1],spikes.shape[2]),compression='gzip',compression_opts=4)

h5File.close()
    




#

sessionId = sessions.index[0]

session = cache.get_ecephys_session(ecephys_session_id=sessionId)

trials = session.trials
stim = session.stimulus_presentations
changeTimes = trials.change_time_no_display_delay[trials.stimulus_change | trials.catch]
flashTimes = stim.start_time[stim.active]

stimTrialIndex = np.searchsorted(trials.start_time,flashTimes) - 1
hasOmission = [any(stim[stim.active].omitted[stimTrialIndex==i]) for i in range(len(trials))]



regions = ('VISp','VISl','VISrl','VISal','VISpm','VISam','LGd','LP','MRN')







