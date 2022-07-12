# -*- coding: utf-8 -*-
"""
Created on Wed Jul  6 13:50:36 2022

@author: svc_ccg
"""

import os
import glob
import time
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
import psytrack



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
        self.abortRate = self.abortedTrials.sum() / self.nTrials
        self.hitRate = self.hit.sum() / (self.hit.sum() + self.miss.sum())
        self.falseAlarmRate = self.falseAlarm.sum() / (self.falseAlarm.sum() + obj.correctReject.sum())
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
        self.firstFlashInTrialStartTimes = np.array([self.flashStartTimes[self.flashTrialIndex==i][0] for i in range(self.nTrials)])
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

goodSessions = ecephys_sessions_table.abnormal_histology.isnull() & ecephys_sessions_table.abnormal_activity.isnull()

novelSessionIds = ecephys_sessions_table.index[goodSessions &
                                               (ecephys_sessions_table.experience_level=='Novel') &
                                               (ecephys_sessions_table.image_set=='H')]


imageSetG = ['im036_r', 'im012_r', 'im044_r', 'im047_r', 'im083_r', 'im111_r', 'im115_r', 'im078_r']
imageSetH = ['im104_r', 'im114_r', 'im083_r', 'im005_r', 'im087_r', 'im024_r', 'im111_r', 'im034_r']
familiarH = np.intersect1d(imageSetH,imageSetG)
novelH = np.setdiff1d(imageSetH,imageSetG)



# load session data
nwbDir = r"D:\visual_behavior_nwbs\visual-behavior-neuropixels-0.1.0\ecephys_sessions"

sessions = []
for i,sid in enumerate(novelSessionIds):
    print('loading session ' + str(i+1) + ' of ' + str(len(novelSessionIds)))
    nwbPath = os.path.join(nwbDir,'ecephys_session_'+str(sid)+'.nwb')
    obj = DocSession(nwbPath)
    sessions.append(obj)


# plots
nHits = [obj.hit.sum() for obj in sessions]
hitRate = [obj.hitRate for obj in sessions]
falseAlarmRate = [obj.falseAlarmRate for obj in sessions]
dprime = [obj.dprime for obj in sessions]
abortRate = [obj.abortRate for obj in sessions]

for d,lbl in zip((nHits,hitRate,falseAlarmRate,dprime),('# hits','hit rate','false alarm rate','d prime')):
    fig = plt.figure()
    ax = plt.subplot(1,1,1)
    ax.plot(abortRate,d,'o',mec='k',mfc='none')
    slope,yint,rval,pval,stderr = scipy.stats.linregress(abortRate,d)
    x = np.array([min(abortRate),max(abortRate)])
    ax.plot(x,slope*x+yint,'--',color='k')
    r,p = scipy.stats.pearsonr(abortRate,d)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xlim([0,1])
    if 'rate' in lbl:
        ax.set_ylim([0,1])
    ax.set_xlabel('abort (early lick) rate')
    ax.set_ylabel(lbl)
    pstr = format(p,'.2E') if p<0.001 else str(round(p,3))
    ax.set_title('r = '+str(round(r,2))+', p = '+pstr,fontsize=10)
    plt.tight_layout()


firstStimDelay = np.concatenate([obj.firstStimDelay for obj in sessions])
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
h,bins = np.histogram(firstStimDelay,bins=np.arange(0,0.75+1/60,1/60))
ax.plot(bins[:-1],h/firstStimDelay.size,'k')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlabel('time from trial start to first stim (s)')
ax.set_ylabel('fraction of trials')
plt.tight_layout()

lickIntervals = np.concatenate([obj.lickIntervals for obj in sessions])
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
h,bins = np.histogram(lickIntervals,bins=np.arange(0,9.75,0.75))
ax.plot(bins[:-1]+0.75/2,h/lickIntervals.size,'k')
ax.set_yscale('log')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlabel('inter-lick interval (s)')
ax.set_ylabel('fraction of licks')
plt.tight_layout()

flashWithLickIntervals = np.concatenate([obj.flashWithLickIntervals for obj in sessions])
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
h,bins = np.histogram(flashWithLickIntervals,bins=np.arange(1,14))
ax.plot(bins[:-1],h/flashWithLickIntervals.size,'k')
ax.set_yscale('log')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlabel('flashes between flashes with licks')
ax.set_ylabel('probability')
plt.tight_layout()
  
flashesToChange = np.concatenate([obj.flashesToChange[obj.changeTrials] for obj in sessions])    
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
h,bins = np.histogram(flashesToChange,bins=np.arange(1,14))
ax.plot(bins[:-1],h/flashesToChange.size,'k')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlabel('flashes to change')
ax.set_ylabel('fraction of change trials')
plt.tight_layout()

flashesToAbort = np.concatenate([obj.flashesToLick[obj.abortedTrials] for obj in sessions])
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
h,bins = np.histogram(flashesToAbort,bins=np.arange(1,14))
ax.plot(bins[:-1],h/flashesToAbort.size,'k')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlabel('flashes to abort')
ax.set_ylabel('fraction of aborted trials')
plt.tight_layout()

flashLickProb = np.stack([np.nanmean(obj.flashLickProb,axis=0) for obj in sessions])
flashLickProbPrevTrialAborted = np.stack([np.nanmean(obj.flashLickProb[np.concatenate(([False],obj.abortedTrials[:-1]))],axis=0) for obj in sessions])
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
x = np.arange(1,13)
ax.plot(x,np.mean(flashLickProb,axis=0),'k',label='all trials')
ax.plot(x,np.mean(flashLickProbPrevTrialAborted,axis=0),'b',label='after aborted trial')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlabel('flashes before abort, omission, or change')
ax.set_ylabel('lick probability')
ax.legend()
plt.tight_layout()

flashLickProbTrials = np.stack([np.sum(~np.isnan(obj.flashLickProb),axis=0) for obj in sessions])
flashLickProbPrevTrialAbortedTrials = np.stack([np.sum(~np.isnan(obj.flashLickProb[np.concatenate(([False],obj.abortedTrials[:-1]))]),axis=0) for obj in sessions])
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(x,np.mean(flashLickProbTrials,axis=0),'k',label='all trials')
ax.plot(x,np.mean(flashLickProbPrevTrialAbortedTrials,axis=0),'b',label='after aborted trial')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlabel('flashes before abort, omission, or change')
ax.set_ylabel('# trials')
ax.legend()
plt.tight_layout()


# fit lick probability curve
x = np.arange(12)
y = np.mean(flashLickProb,axis=0)
bounds = ((0,0,0,0,0,0,0),(1,1,12,1,1,12,12))
fitParams = fitCurve(expDecayPlusGauss,x,y,bounds=bounds)
ampExp,offsetExp,tau,ampGauss,offsetGauss,mu,sigma = fitParams
fity = expDecayPlusGauss(x,*fitParams)

impulsivityLickProb = expDecay(x,ampExp,offsetExp,tau)
timingLickProb = gauss(x,ampGauss,offsetGauss,mu,sigma)

impulsivityLickProbNorm = impulsivityLickProb / impulsivityLickProb.max()
timingLickProbNorm = timingLickProb / timingLickProb.max()

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

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
x = np.arange(1,13)
ax.plot(x,impulsivityLickProb,'k',label='impulsivity')
ax.plot(x,timingLickProb,'b',label='timing')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlabel('flashes before abort, omission, or change')
ax.set_ylabel('lick probability')
ax.legend()
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
x = np.arange(1,13)
ax.plot(x,impulsivityLickProbNorm,'k',label='impulsivity')
ax.plot(x,timingLickProbNorm,'b',label='timing')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlabel('flashes before abort, omission, or change')
ax.set_ylabel('normalized lick probability')
ax.legend()
plt.tight_layout()


# psytrack

# exclude flashes after trial outcome decided and ignore autorewarded trials
regressors = ['impulsivity','timing','omission','change','novelty']
x = {r: [] for r in regressors}
y = []
for obj in sessions:
    impulsivity = np.zeros(obj.nFlashes)
    timing = impulsivity.copy()
    flashHasLick = obj.flashHasLick.astype(float)
    for trialInd in range(obj.nTrials):
        flashInd = np.where(obj.flashTrialIndex==trialInd)[0]
        if flashInd.size < impulsivityLickProbNorm.size:
            impulsivity[flashInd] = impulsivityLickProbNorm[:flashInd.size]
        else:
            impulsivity[flashInd[:impulsivityLickProbNorm.size]] = impulsivityLickProbNorm
        if flashInd.size < timingLickProbNorm.size:
            timing[flashInd] = timingLickProbNorm[:flashInd.size]
        else:
            timing[flashInd[:timingLickProbNorm.size]] = timingLickProbNorm
        if obj.autoRewarded[trialInd]:
            flashHasLick[flashInd] = np.nan
        elif obj.abortedTrials[trialInd]:
            lickFlash = np.where(obj.flashHasLick[flashInd])[0]
            if len(lickFlash)>0:
                flashHasLick[flashInd[lickFlash[0]+1:]] = np.nan
        elif obj.changeTrials[trialInd] or obj.catchTrials[trialInd]:
            changeFlash = np.where(obj.flashIsChange[flashInd] | obj.flashIsCatch[flashInd])[0]
            if len(changeFlash)>0:
                flashHasLick[flashInd[changeFlash[0]+1:]] = np.nan
    x['impulsivity'].append(impulsivity)
    x['timing'].append(timing)
    postOmitted = np.where(obj.flashOmitted)[0] + 1
    omission = np.zeros(obj.nFlashes)
    omission[postOmitted[postOmitted<omission.size]]=True
    x['omission'].append(omission)
    x['change'].append(obj.flashIsChange.astype(float))
    x['novelty'].append(np.in1d(obj.flashImage,novelH).astype(float))
    y.append(flashHasLick)
        
excludedFlashes = [np.isnan(a) for a in y]

d = {'inputs': {key: np.concatenate(val)[~np.concatenate(excludedFlashes)][:,None] for key,val in x.items()},
     'y': np.concatenate(y)[~np.concatenate(excludedFlashes)],
     'dayLength': np.array([np.sum(~b) for b in excludedFlashes])}

weights = {key: 1 for key in d['inputs']}

nWeights = sum(weights.values())

hyper= {'sigInit': 2**4.,
        'sigma': [2**-4.] * nWeights,
        'sigDay': [2**-4.] * nWeights}

optList = ['sigma','sigDay']

t = time.perf_counter()
hyp, evd, wMode, hess_info = psytrack.hyperOpt(d, hyper, weights, optList)
print(time.perf_counter()-t)

cvFolds = 5
cvTrials = d['y'].size - (d['y'].size % cvFolds)
t = time.perf_counter()
cvLikelihood,cvProbMiss = psytrack.crossValidate(psytrack.trim(d,END=cvTrials), hyper, weights, optList, F=cvFolds, seed=0)
print(time.perf_counter()-t)
cvProbResp = 1-cvProbMiss

yModel = (cvProbResp>=0.5).astype(float)
accuracy = 1 - (np.abs(y[:cvTrials]-yModel).sum() / cvTrials)





















