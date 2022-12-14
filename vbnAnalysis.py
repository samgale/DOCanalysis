# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 16:22:30 2022

@author: svc_ccg
"""

import numpy as np
from allensdk.brain_observatory.behavior.behavior_project_cache.behavior_neuropixels_project_cache import VisualBehaviorNeuropixelsProjectCache


def makePSTH(spikes, startTimes, windowDur, binSize=0.001):
    bins = np.arange(0,windowDur+binSize,binSize)
    counts = np.zeros(bins.size-1)
    for start in startTimes:
        startInd = np.searchsorted(spikes, start)
        endInd = np.searchsorted(spikes, start+windowDur)
        counts = counts + np.histogram(spikes[startInd:endInd]-start, bins)[0]
    counts = counts/len(startTimes)
    return counts/binSize, bins[:-1]

vbnCache = r'\\allen\aibs\informatics\chris.morrison\ticket-27\allensdk_caches\vbn_cache_2022_Jul29'

cache = VisualBehaviorNeuropixelsProjectCache.from_local_cache(cache_dir=vbnCache)

sessions = cache.get_ecephys_session_table(filter_abnormalities=False)

sessionId = sessions.index[0]

session = cache.get_ecephys_session(ecephys_session_id=sessionId)

trials = session.trials
stim = session.stimulus_presentations

stimTrialIndex = np.searchsorted(trials.start_time,stim.start_time[stim.active]) - 1
hasOmission = [any(stim[stim.active].omitted[stimTrialIndex==i]) for i in range(len(trials))]

units = session.get_units()
channels = session.get_channels()
units = units.merge(channels,left_on='peak_channel_id',right_index=True)
goodUnits = units[(units['quality']=='good') & (units['snr']>1) & (units['isi_violations']<1)]
spikeTimes = session.spike_times

regions = ('VISp','VISl','VISrl','VISal','VISpm','VISam','LGd','LP','MRN')




















