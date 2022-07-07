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

nwbPath = r"D:\visual_behavior_nwbs\ecephys_session_1043752325.nwb"


with NWBHDF5IO(nwbPath, 'r', load_namespaces=True) as nwb_io:
    session = BehaviorEcephysSession.from_nwb(nwbfile=nwb_io.read())

sessionID = session.metadata['ecephys_session_id']
trials = session.trials
stim = session.stimulus_presentations
licks = session.licks


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


flashLickProb = np.zeros((trials.shape[0],12))
flashLickProb[~noLickAbortedTrials,trialFlashes[~noLickAbortedTrials]-1] = True

flashLickProb[~ignoreTrials & abortedTrials].mean(axis=0)


