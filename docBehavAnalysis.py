# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 13:50:36 2022

@author: svc_ccg
"""

import os
import glob
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
from pynwb import NWBHDF5IO
from allensdk.brain_observatory.ecephys.behavior_ecephys_session import (
    BehaviorEcephysSession)
import allensdk
from allensdk.brain_observatory.behavior.behavior_project_cache.\
    behavior_neuropixels_project_cache \
    import VisualBehaviorNeuropixelsProjectCache



def fitCurve(func,x,y,initGuess=None,bounds=None):
    return scipy.optimize.curve_fit(func,x,y,p0=initGuess,bounds=bounds)[0]
    

def expDecay(x,amp,offset,tau):
    return amp * np.exp(-x/tau) + offset


def gauss(x,amp,offset,mu,sigma):
    return amp * np.exp((-(x-mu)**2) / (2*(sigma**2))) + offset


def expDecayPlusGauss(x,ampExp,offsetExp,tau,ampGauss,offsetGauss,mu,sigma):
    return expDecay(x,ampExp,offsetExp,tau) + gauss(x,ampGauss,offsetGauss,mu,sigma)


def calcDprime(hits,misses,falseAlarms,correctRejects):
    hitRate = calcHitRate(hits,misses,adjusted=True)
    falseAlarmRate = calcHitRate(falseAlarms,correctRejects,adjusted=True)
    z = [scipy.stats.norm.ppf(r) for r in (hitRate,falseAlarmRate)]
    return z[0]-z[1]


def calcHitRate(hits,misses,adjusted=False):
    n = hits+misses
    if n==0:
        return np.nan
    hitRate = hits/n
    if adjusted:
        if hitRate==0:
            hitRate = 0.5/n
        elif hitRate==1:
            hitRate = 1-0.5/n
    return hitRate



class DocSession():
    
    def __init__(self,nwbPath):
        self.nwbPath = nwbPath
        self.loadData()
        
    def loadData(self):
        with NWBHDF5IO(nwbPath, 'r', load_namespaces=True) as nwb_io:
            session = BehaviorEcephysSession.from_nwb(nwbfile=nwb_io.read())
        self.sessionID = session.metadata['ecephys_session_id']

        trials = session.trials
        stim = session.stimulus_presentations
        licks = session.licks
        
        # trial data
        self.nTrials = trials.shape[0]
        self.trialStartTimes = np.array(trials.start_time)
        self.trialChangeTimes = np.array(trials.change_time_no_display_delay)
        self.initialImage = np.array(trials.initial_image_name)
        self.changeImage = np.array(trials.change_image_name)
        self.changeTrials = np.array(trials.stimulus_change)
        self.catchTrials = np.array(trials.catch)
        self.abortedTrials = np.array(trials.aborted)
        self.autoRewarded = np.array(trials.auto_rewarded)
        self.hit = np.array(trials.hit)
        self.miss = np.array(trials.miss)
        self.falseAlarm = np.array(trials.false_alarm)
        self.correctReject = np.array(trials.correct_reject)
        self.dprime = calcDprime(self.hit.sum(),self.miss.sum(),self.falseAlarm.sum(),self.correctReject.sum())
        
        # flash data
        self.nFlashes = np.sum(stim.active)
        self.flashStartTimes = np.array(stim.start_time[stim.active])
        self.flashImage = np.array(stim.image_name[stim.active])
        self.flashOmitted = np.array(stim.omitted[stim.active])
        self.flashIsChange = np.array(stim.is_change[stim.active])
        self.flashIsCatch = np.zeros(self.nFlashes,dtype=bool)
        self.flashIsCatch[np.searchsorted(self.flashStartTimes,self.trialChangeTimes[self.catchTrials])] = True
        
        # get index of corresponding trial for each flash
        self.flashTrialIndex = np.searchsorted(self.trialStartTimes,self.flashStartTimes) - 1
        
        # calculate delay between trial start times and first flash start times
        self.firstFlashInTrialStartTimes = np.array([self.flashStartTimes[self.flashTrialIndex==i] for i in range(self.nTrials)])
        self.firstStimDelay = self.firstFlashInTrialStartTimes - self.trialStartTimes
        
        # lick data
        self.lickTimes = np.array(licks['timestamps'])
        self.lickIntervals = np.diff(self.lickTimes[self.lickTimes<trials.stop_time.iloc[self.nTrials-1]])
        
        # find flashes with licks
        minLickLatency = 0.15
        self.flashHasLick = np.zeros(self.nFlashes)
        startStop = np.concatenate((self.flashStartTimes,[trials.stop_time.iloc[-1]]))
        for i in range(self.nFlashes):
            if np.any((self.lickTimes >= startStop[i]+minLickLatency) & (self.lickTimes < startStop[i+1]+minLickLatency)):
                self.flashHasLick[i] = True

        self.flashesToLick = np.full(self.nTrials,np.nan)
        self.flashesToChange = self.flashesToLick.copy()
        self.flashesToOmitted = self.flashesToLick.copy()
        for trialInd in range(self.nTrials):
            flashInd = np.where(self.flashTrialIndex==trialInd)[0]
            lickFlash = np.where(self.flashHasLick[flashInd])[0]
            if len(lickFlash)>0:
                self.flashesToLick[trialInd] = lickFlash[0]+1
            changeFlash = np.where(self.flashIsChange[flashInd])[0]
            if len(changeFlash)>0:
                self.flashesToChange[trialInd] = changeFlash[0]+1
            omittedFlash = np.where(self.flashOmitted[flashInd])[0]
            if len(omittedFlash)>0:
                self.flashesToOmitted[trialInd] = omittedFlash[0]
        
        self.flashWithLickIntervals = np.diff(np.where(self.flashHasLick)[0])
                
        # find lick probability for each flash excluding flashes after abort, change, or omission 
        self.flashLickProb = np.zeros((self.nTrials,12))
        for i,(j,k,l) in enumerate(zip(self.flashesToLick,self.flashesToChange,self.flashesToOmitted)):
            if j<12:
                j = int(j)
                self.flashLickProb[i,j-1] = 1
                self.flashLickProb[i,j:] = np.nan
            if not np.isnan(k):
                self.flashLickProb[i,int(k)-1:] = np.nan
            if not np.isnan(l):
                self.flashLickProb[i,int(l)-1:] = np.nan



# get novel session IDs
cache_dir = r"D:\allensdk_cache"

cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(
            cache_dir=cache_dir)


ecephys_sessions_table = cache.get_ecephys_session_table()[0]
ecephys_sessions_table.columns

novelSessionIds = ecephys_sessions_table.index[(ecephys_sessions_table.experience_level=='Novel') &
                                               (ecephys_sessions_table.image_set=='H')]



# load session data
nwbDir = r"D:\visual_behavior_nwbs\visual-behavior-neuropixels-0.1.0\ecephys_sessions"
sessionIds = ['1043752325']

sessions = []
for sid in sessionIds:
    nwbPath = os.path.join(nwbDir,'ecephys_session_'+sid+'.nwb')
    obj = DocSession(nwbPath)
    sessions.append(obj)


# summary plots for individual session
obj = sessions[0]

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.hist(obj.firstStimDelay,color='k')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlabel('time from trial start to first stim (s)')
ax.set_ylabel('# trials')
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
h,bins = np.histogram(obj.ickIntervals,bins=np.arange(0,9.75,0.75))
ax.plot(bins[:-1]+0.75/2,h/obj.lickIntervals.size,'k')
ax.set_yscale('log')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlabel('inter-lick interval (s)')
ax.set_ylabel('fraction of licks')
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
h,bins = np.histogram(obj.flashWithLickIntervals,bins=np.arange(1,14))
ax.plot(bins[:-1],h/obj.flashWithLickIntervals.size,'k')
ax.set_yscale('log')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlabel('flashes between flashes with licks')
ax.set_ylabel('probability')
plt.tight_layout()
      
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
h,bins = np.histogram(obj.flashesToChange[obj.changeTrials],bins=np.arange(1,14))
ax.plot(bins[:-1],h/np.sum(obj.changeTrials),'k')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlabel('flashes to change')
ax.set_ylabel('fraction of change trials')
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
h,bins = np.histogram(obj.flashesToLick[obj.abortedTrials],bins=np.arange(1,14))
ax.plot(bins[:-1],h/np.sum(obj.abortedTrials),'k')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlabel('flashes to abort')
ax.set_ylabel('fraction of aborted trials')
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
x = np.arange(1,13)
prevTrialAborted = np.concatenate(([False],obj.abortedTrials[:-1]))
ax.plot(x,np.nanmean(obj.flashLickProb,axis=0),'k',label='all trials')
ax.plot(x,np.nanmean(obj.flashLickProb[prevTrialAborted],axis=0),'b',label='after aborted trial')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlabel('flashes before abort, omission, or change')
ax.set_ylabel('lick probability')
ax.legend()
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(x,np.sum(~np.isnan(obj.flashLickProb),axis=0),'k',label='all trials')
ax.plot(x,np.sum(~np.isnan(obj.flashLickProb[prevTrialAborted]),axis=0),'b',label='after aborted trial')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlabel('flashes before abort, omission, or change')
ax.set_ylabel('# trials')
ax.legend()
plt.tight_layout()


# fit lick probability curve
x = np.arange(12)
y = np.nanmean(obj.flashLickProb,axis=0)
bounds = ((0,0,0,0,0,0,0),(1,1,12,1,1,12,12))
fitParams = fitCurve(expDecayPlusGauss,x,y,bounds=bounds)
ampExp,offsetExp,tau,ampGauss,offsetGauss,mu,sigma = fitParams
fity = expDecayPlusGauss(x,*fitParams)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
x = np.arange(1,13)
ax.plot(x,y,'k',label='observed')
ax.plot(x,fity,'r',label='fit:'+'\nexp amp = '+str(round(ampExp,2))+'\ngauss amp = '+str(round(ampGauss,2))+'\ngauss mean = '+str(round(mu+1,2)))
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlabel('flashes before abort, omission, or change')
ax.set_ylabel('lick probability')
ax.legend()
plt.tight_layout()

impulsivityLickProb = expDecay(x,ampExp,offsetExp,tau)
impulsivityLickProb /= impulsivityLickProb.max()
timingLickProb = gauss(x,ampGauss,offsetGauss,mu,sigma)
timingLickProb /= timingLickProb.max()




# psytrack

# exclude flashes after trial outcome decided
self.flashesToLick = np.full(self.nTrials,np.nan)
self.flashesToChange = self.flashesToLick.copy()
self.flashesToOmitted = self.flashesToLick.copy()
for trialInd in range(self.nTrials):
    flashInd = np.where(self.flashTrialIndex==trialInd)[0]
    lickFlash = np.where(self.flashHasLick[flashInd])[0]
    if len(lickFlash)>0:
        self.flashesToLick[trialInd] = lickFlash[0]+1
    changeFlash = np.where(self.flashIsChange[flashInd])[0]
    if len(changeFlash)>0:
        self.flashesToChange[trialInd] = changeFlash[0]+1
    omittedFlash = np.where(self.flashOmitted[flashInd])[0]
    if len(omittedFlash)>0:
        self.flashesToOmitted[trialInd] = omittedFlash[0]
        
flashHasLick = obj.flashHasLick.copy()
for i,(j,k,l) in enumerate(zip(self.flashesToLick,self.flashesToChange,self.flashesToOmitted)):
    if j<12:
        j = int(j)
        self.flashLickProb[i,j-1] = 1
        self.flashLickProb[i,j:] = np.nan
    if not np.isnan(k):
        self.flashLickProb[i,int(k)-1:] = np.nan
    if not np.isnan(l):
        self.flashLickProb[i,int(l)-1:] = np.nan

regressors = ['impulsivity','timing','omission','change','autoreward']
y = np.concatenate([obj.trialResponse for obj in exps]).astype(float)
x = np.zeros((y.size,len(regressors)))
for i,stim in enumerate(np.concatenate([obj.trialStim for obj in exps])):
    if stim in regressors:
        x[i,regressors.index(stim)] = 1
x[:,-1] = 1

autoRewarded = np.concatenate([obj.autoRewarded for obj in exps])
x = x[~autoRewarded]
y = y[~autoRewarded]

sessionTrials = [obj.nTrials-obj.autoRewarded.sum() for obj in exps]
sessionStartStop = np.concatenate(([0],np.cumsum(sessionTrials)))

sessionBlockTrials = [[np.sum(obj.trialBlock[~obj.autoRewarded]==i) for i in np.unique(obj.trialBlock)] for obj in exps]
blockTrials = np.concatenate(sessionBlockTrials)



# psytrack
d = {'inputs': {key: val[:,None] for key,val in zip(regressors,x.T)},
     'y': y.copy(),
     'dayLength': blockTrials}

weights = {key: 1 for key in d['inputs']}

nWeights = sum(weights.values())

hyper= {'sigInit': 2**4.,
        'sigma': [2**-4.] * nWeights,
        'sigDay': [2**-4.] * nWeights}

optList = ['sigma','sigDay']

hyp, evd, wMode, hess_info = psytrack.hyperOpt(d, hyper, weights, optList)

cvFolds = 5
cvTrials = y.size - (y.size % cvFolds)
cvLikelihood,cvProbMiss = psytrack.crossValidate(psytrack.trim(d,END=cvTrials), hyper, weights, optList, F=cvFolds, seed=0)
cvProbResp = 1-cvProbMiss

yModel = (cvProbResp>=0.5).astype(float)
accuracy = 1 - (np.abs(y[:cvTrials]-yModel).sum() / cvTrials)





















