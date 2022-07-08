# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 13:50:36 2022

@author: svc_ccg
"""

import os
import glob
import numpy as np
import scipy.signal
import scipy.ndimage
import matplotlib.pyplot as plt
from pynwb import NWBHDF5IO
from allensdk.brain_observatory.ecephys.behavior_ecephys_session import (
    BehaviorEcephysSession)


nwb_base = r"\\allen\programs\mindscope\workgroups\np-behavior\vbn_data_release\vbn_s3_cache\visual-behavior-neuropixels-0.1.0\ecephys_sessions"
nwb_paths = glob.glob(os.path.join(nwb_base, '*nwb'))


nwbPath = nwb_paths[0]

nwbPath = r"D:\visual_behavior_nwbs\ecephys_session_1044597824.nwb"


with NWBHDF5IO(nwbPath, 'r', load_namespaces=True) as nwb_io:
    session = BehaviorEcephysSession.from_nwb(nwbfile=nwb_io.read())

sessionID = session.metadata['ecephys_session_id']
trials = session.trials
stim = session.stimulus_presentations
licks = session.licks
lickTimes = np.array(licks['timestamps'])

nTrials = trials.shape[0]
nStim = np.sum(stim.active)

# get index of corresponding trial for each stim in stim table
stimTrialIndex = np.searchsorted(trials.start_time,stim.start_time[stim.active]) - 1

# # check delay between trial start times and first stim start times
# firstStimInTrialStartTimes = np.array([stim.start_time[stim.active][stimTrialIndex==i].iloc[0] for i in range(nTrials)])
# firstStimDelay = firstStimInTrialStartTimes - trials.start_time

# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
# ax.hist(firstStimDelay)
# ax.set_xlabel('time from trial start to first stim (s)')
# ax.set_ylabel('# trials')

# # use the trial indices to re-create the image sequence
# # and compare to the image sequence in the stim table (ignoring omitted flashes)
# trialImageSequence = np.empty(numStim,dtype=object)
# for trialInd,(initialImage,changeImage) in enumerate(zip(trials.initial_image_name,trials.change_image_name)):
#     stimInd = np.where(stimTrialIndex==trialInd)[0]
#     trialStart,trialEnd = stimInd[[0,-1]]
#     changeInd = np.where(stim.is_change[stimInd])[0]
#     changeInd = stimInd[changeInd[0]] if len(changeInd) > 0 else trialEnd+1
#     trialImageSequence[trialStart:changeInd] = initialImage
#     trialImageSequence[changeInd:trialEnd+1] = changeImage

# stimImageSequence = np.array(stim.image_name[stim.active])
# omitted = np.array(stim.omitted[stim.active]).astype(bool)
# assert(np.all(trialImageSequence[~omitted]==stimImageSequence[~omitted]))



flashHasLick = np.zeros(nStim)
stimStart = np.concatenate((stim.start_time[stim.active],[trials.stop_time.iloc[-1]]))
for i in range(nStim):
    if np.any((lickTimes >= stimStart[i]) & (lickTimes < stimStart[i+1])):
        flashHasLick[i] = True

flashesToLick = np.full(trials.shape[0],np.nan)
flashesToChange = flashesToLick.copy()
for trialInd in range(nTrials):
    stimInd = np.where(stimTrialIndex==trialInd)[0]
    lickFlash = np.where(flashHasLick[stimInd])[0]
    if len(lickFlash)>0:
        flashesToLick[trialInd] = lickFlash[0]+1
    stimFlash = np.where(stim.is_change[stimInd])[0]
    if len(stimFlash)>0:
        flashesToChange[trialInd] = stimFlash[0]+1
        

flashesToHit = flashesToLick[trials.hit]
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
h,bins = np.histogram(flashesToHit,bins=np.arange(0,flashesToHit.max()+3))
ax.plot(bins[:-1],h,'k')
ax.set_xlabel('flashes to hit')
ax.set_ylabel('trials')




flashLickProb = np.zeros((trials.shape[0],12))
for i,(j,k) in enumerate(zip(flashesToLick,flashesToChange)):
    if j<12:
        j = int(j)
        flashLickProb[i,j-1] = 1
        flashLickProb[i,j:] = np.nan
    if not np.isnan(k):
        flashLickProb[i,int(k)-1:] = np.nan

prevAborted = np.concatenate(([False],trials.aborted[:-1]))
plt.plot(np.nanmean(flashLickProb[prevAborted],axis=0),'k')



abortedTrials = np.array(trials.aborted)
noLickAbortedTrials = abortedTrials & np.array([len(lt)<1 for lt in trials.lick_times])
autoRewardedTrials = np.array(trials.auto_rewarded)
ignoreTrials = noLickAbortedTrials | autoRewardedTrials


trialStartTime= np.array(trials.start_time)
trialEndTime = np.array(trials.change_time_no_display_delay)

ind = abortedTrials & ~ignoreTrials
trialEndTime[ind] = np.array([lt[0] for lt in trials.lick_times[ind]])

trialStartFlash,trialEndFlash = [np.searchsorted(stim.start_time[stim.active],st) for st in (trialStartTime,trialEndTime)]

trialFlashes = trialEndFlash-trialStartFlash+1



























