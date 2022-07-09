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


def fitCurve(func,x,y,initGuess=None,bounds=None):
    return scipy.optimize.curve_fit(func,x,y,p0=initGuess,bounds=bounds)[0]
    

def expDecay(x,amp,offset,tau):
    return amp * np.exp(-x/tau) + offset


def gauss(x,amp,offset,mu,sigma):
    return amp * np.exp((-(x-mu)**2) / (2*(sigma**2))) + offset


def expDecayPlusGauss(x,ampExp,offsetExp,tau,ampGauss,offsetGauss,mu,sigma):
    return expDecay(x,ampExp,offsetExp,tau) + gauss(x,ampGauss,offsetGauss,mu,sigma)



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

# check delay between trial start times and first stim start times
firstStimInTrialStartTimes = np.array([stim.start_time[stim.active][stimTrialIndex==i].iloc[0] for i in range(nTrials)])
firstStimDelay = firstStimInTrialStartTimes - trials.start_time

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.hist(firstStimDelay,color='k')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlabel('time from trial start to first stim (s)')
ax.set_ylabel('# trials')
plt.tight_layout()


# find flashes with licks
minLickLatency = 0.15
flashHasLick = np.zeros(nStim)
stimStart = np.concatenate((stim.start_time[stim.active],[trials.stop_time.iloc[-1]]))
for i in range(nStim):
    if np.any((lickTimes >= stimStart[i]+minLickLatency) & (lickTimes < stimStart[i+1]+minLickLatency)):
        flashHasLick[i] = True

flashesToLick = np.full(trials.shape[0],np.nan)
flashesToChange = flashesToLick.copy()
flashesToOmitted = flashesToLick.copy()
for trialInd in range(nTrials):
    stimInd = np.where(stimTrialIndex==trialInd)[0]
    lickFlash = np.where(flashHasLick[stimInd])[0]
    if len(lickFlash)>0:
        flashesToLick[trialInd] = lickFlash[0]+1
    changeFlash = np.where(stim.is_change[stimInd])[0]
    if len(changeFlash)>0:
        flashesToChange[trialInd] = changeFlash[0]+1
    omittedFlash = np.where(stim.omitted[stimInd])[0]
    if len(omittedFlash)>0:
        flashesToOmitted[trialInd] = omittedFlash[0]
        
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
h,bins = np.histogram(flashesToChange[trials.stimulus_change],bins=np.arange(1,14))
ax.plot(bins[:-1],h/np.sum(trials.stimulus_change),'k')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlabel('flashes to change')
ax.set_ylabel('fraction of change trials')
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
h,bins = np.histogram(flashesToLick[trials.aborted],bins=np.arange(1,14))
ax.plot(bins[:-1],h/np.sum(trials.aborted),'k')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlabel('flashes to abort')
ax.set_ylabel('fraction of aborted trials')
plt.tight_layout()


# find lick probability for each flash excluding flashes after abort, change, or omission 
flashLickProb = np.zeros((trials.shape[0],12))
for i,(j,k,l) in enumerate(zip(flashesToLick,flashesToChange,flashesToOmitted)):
    if j<12:
        j = int(j)
        flashLickProb[i,j-1] = 1
        flashLickProb[i,j:] = np.nan
    if not np.isnan(k):
        flashLickProb[i,int(k)-1:] = np.nan
    if not np.isnan(l):
        flashLickProb[i,int(l)-1:] = np.nan

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
x = np.arange(1,13)
prevTrialAborted = np.concatenate(([False],trials.aborted[:-1]))
ax.plot(x,np.nanmean(flashLickProb,axis=0),'k',label='all trials')
ax.plot(x,np.nanmean(flashLickProb[prevTrialAborted],axis=0),'b',label='after aborted trial')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlabel('flashes before abort, omission, or change')
ax.set_ylabel('lick probability')
ax.legend()
plt.tight_layout()


# fit lick probability curve
x = np.arange(12)
y = np.nanmean(flashLickProb,axis=0)
bounds = ((0,0,0,0,0,0,0),(1,1,12,1,1,12,12))
fitParams = fitCurve(expDecayPlusGauss,x,y,bounds=bounds)
ampExp,offsetExp,tau,ampGauss,offsetGauss,mu,sigma = fitParams
fity = expDecayPlusGauss(x,*fitParams)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
x = np.arange(1,13)
prevTrialAborted = np.concatenate(([False],trials.aborted[:-1]))
ax.plot(x,y,'k',label='observed')
ax.plot(x,fity,'r',label='fit:'+'\nexp amp = '+str(round(ampExp,2))+'\ngauss amp = '+str(round(ampGauss,2))+'\ngauss mean = '+str(round(mu+1,2)))
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlabel('flashes before abort, omission, or change')
ax.set_ylabel('lick probability')
ax.legend()
plt.tight_layout()


# lick intervals
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
lickIntervals = np.diff(lickTimes[lickTimes<trials.stop_time.iloc[nTrials-1]])
h,bins = np.histogram(lickIntervals,bins=np.arange(0,9.75,0.75))
ax.plot(bins[:-1]+0.75/2,h/lickIntervals.size,'k')
ax.set_yscale('log')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlabel('inter-lick interval (s)')
ax.set_ylabel('fraction of licks')
plt.tight_layout()


# intervals (number of flashes) between flashes with licks
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
flashWithLickIntervals = np.diff(np.where(flashHasLick)[0])
h,bins = np.histogram(flashWithLickIntervals,bins=np.arange(1,14))
ax.plot(bins[:-1],h/flashWithLickIntervals.size,'k')
ax.set_yscale('log')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlabel('flashes between flashes with licks')
ax.set_ylabel('probability')
plt.tight_layout()























