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


def gamma(x,amp,offset,shape,loc):
    return amp * scipy.stats.gamma.pdf(x,shape,loc) + offset


def expDecayPlusGauss(x,ampExp,offsetExp,tau,ampGauss,offsetGauss,mu,sigma):
    return expDecay(x,ampExp,offsetExp,tau) + gauss(x,ampGauss,offsetGauss,mu,sigma)


def expDecayPlusGamma(x,ampExp,offsetExp,tau,ampGamma,offsetGamma,shape,loc):
    return expDecay(x,ampExp,offsetExp,tau) + gamma(x,ampGamma,offsetGamma,shape,loc)


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
        self.trialStopTimes = np.array(trials.stop_time)
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
        self.falseAlarmRate = self.falseAlarm.sum() / (self.falseAlarm.sum() + self.correctReject.sum())
        self.dprime = calcDprime(self.hit.sum(),self.miss.sum(),self.falseAlarm.sum(),self.correctReject.sum())
        
        # flash data
        self.nFlashes = np.sum(stim.active)
        self.flashStartTimes = np.array(stim.start_time[stim.active])
        self.flashStopTimes = np.array(stim.stop_time[stim.active])
        self.flashImage = np.array(stim.image_name[stim.active])
        self.flashOmitted = np.array(stim.omitted[stim.active]).astype(bool)
        self.flashIsChange = np.array(stim.is_change[stim.active]).astype(bool)
        self.flashIsCatch = np.zeros(self.nFlashes,dtype=bool)
        self.flashIsCatch[np.searchsorted(self.flashStartTimes,self.trialChangeTimes[self.catchTrials])] = True
        
        # lick data
        self.lickTimes = np.array(licks['timestamps'])
        self.lickIntervals = np.diff(self.lickTimes[self.lickTimes<self.trialStopTimes[-1]])
        
        # get index of corresponding trial and trial outcome for each flash
        self.flashTrialIndex = np.searchsorted(self.trialStartTimes,self.flashStartTimes) - 1
        self.flashTrialOutcome = np.empty(self.nFlashes,dtype='O')
        for ind,outcome in zip((self.hit,self.miss,self.falseAlarm,self.correctReject,self.autoRewarded,self.abortedTrials),
                               ('hit','miss','false alarm','correct reject','autoreward','abort')):
            self.flashTrialOutcome[np.in1d(self.flashTrialIndex,np.where(ind)[0])] = outcome
            
        # calculate delay between trial start times and first flash start times
        self.firstFlashInTrialStartTimes = np.array([self.flashStartTimes[self.flashTrialIndex==i][0] for i in range(self.nTrials)])
        self.firstStimDelay = self.firstFlashInTrialStartTimes - self.trialStartTimes
        
        # find trials with omitted flashes
        self.omittedFlashTrials = np.zeros(self.nTrials,dtype=bool)
        self.omittedFlashTrials[self.flashTrialIndex[self.flashOmitted]] = True
        
        # find flashes causing aborts
        self.flashIsAbort = np.zeros(self.nFlashes,dtype=bool)
        self.noLickAbortedTrials = np.ones(self.nTrials,dtype=bool)
        for trialInd in np.where(self.abortedTrials)[0]:
            for i in np.where(self.flashTrialIndex==trialInd)[0]:
                stopTime = self.flashStartTimes[i+1] if i+1 < self.nFlashes else self.flashStartTimes[i] + 0.75
                if np.any((self.lickTimes >= self.flashStartTimes[i]) & (self.lickTimes < stopTime)):
                    self.flashIsAbort[i] = True
                    self.noLickAbortedTrials[trialInd] = False
                    break
        
    def getLickProb(self):
        # find flashes with licks and lick latency
        minLickLatency = 0.15
        self.flashHasLick = np.zeros(self.nFlashes,dtype=bool)
        self.flashLickLatency = np.full(self.nFlashes,np.nan)
        for i in range(self.nFlashes):
            stopTime = self.flashStartTimes[i+1] if i+1 < self.nFlashes else self.flashStartTimes[i] + 0.75
            lt = (self.lickTimes >= self.flashStartTimes[i] + minLickLatency) & (self.lickTimes < stopTime + minLickLatency)
            if np.any(lt):
                self.flashHasLick[i] = True
                self.flashLickLatency[i] = self.lickTimes[lt][0] - self.flashStartTimes[i]        
        self.flashWithLickIntervals = np.diff(np.where(self.flashHasLick)[0])
        
        self.flashesToLick = np.full(self.nTrials,np.nan)
        self.flashesToChange = self.flashesToLick.copy()
        self.flashesToOmitted = self.flashesToLick.copy()
        self.omittedFlashLickProb = np.full((self.nTrials,12),np.nan)
        self.postOmittedFlashLickProb = self.omittedFlashLickProb.copy()
        self.omittedFlashLickLat = np.full(self.nTrials,np.nan)
        self.postOmittedFlashLickLat = self.omittedFlashLickLat.copy()
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
                self.flashesToOmitted[trialInd] = omittedFlash[0]+1
                if omittedFlash[0]<12:
                    ind = flashInd[omittedFlash[0]]
                    self.omittedFlashLickProb[trialInd,omittedFlash[0]] = self.flashHasLick[ind]
                    self.omittedFlashLickLat[trialInd] = self.flashLickLatency[ind]
                    if ind+1<obj.nFlashes:
                        self.postOmittedFlashLickProb[trialInd,omittedFlash[0]] = self.flashHasLick[ind+1]
                        self.postOmittedFlashLickLat[trialInd] = self.flashLickLatency[ind+1]
        
        # find lick probability for each flash excluding flashes after abort, change, or omission 
        self.flashLickProb = np.zeros((self.nTrials,12))
        for i,(j,k,l) in enumerate(zip(self.flashesToLick,self.flashesToChange,self.flashesToOmitted)):
            if j<13:
                self.flashLickProb[i,int(j)-1] = 1
                self.flashLickProb[i,int(j):] = np.nan
            if not np.isnan(k):
                self.flashLickProb[i,int(k)-1:] = np.nan
            if not np.isnan(l):
                self.flashLickProb[i,int(l)-1:] = np.nan



# get novel session IDs
cache_dir = r"D:\allensdk_cache"

cache = VisualBehaviorNeuropixelsProjectCache.from_s3_cache(cache_dir=cache_dir)

sessionInfo = cache.get_ecephys_session_table()[0]
sessionInfo.columns

notAbnormalSessions = np.array(sessionInfo.abnormal_histology.isnull() & sessionInfo.abnormal_activity.isnull())

imageSetG = ['im083_r', 'im111_r', 'im036_r', 'im012_r', 'im044_r', 'im047_r', 'im115_r', 'im078_r']
imageSetH = ['im083_r', 'im111_r', 'im104_r', 'im114_r', 'im005_r', 'im087_r', 'im024_r', 'im034_r']
familiarImages = np.intersect1d(imageSetH,imageSetG)
novelImagesG = np.setdiff1d(imageSetG,imageSetH)
novelImagesH = np.setdiff1d(imageSetH,imageSetG)



# load session data
nwbDir = r"D:\visual_behavior_nwbs\visual-behavior-neuropixels-0.1.0\ecephys_sessions"

sessionIds = sessionInfo.index[notAbnormalSessions]
allSessions = []
for i,sid in enumerate(sessionIds):
    print('loading session ' + str(i+1) + ' of ' + str(len(sessionIds)))
    nwbPath = os.path.join(nwbDir,'ecephys_session_'+str(sid)+'.nwb')
    obj = DocSession(nwbPath)
    obj.getLickProb()
    obj.day = sessionInfo.loc[sid].session_number
    obj.novel = True if sessionInfo.loc[sid].experience_level=='Novel' else False
    obj.imageSet = sessionInfo.loc[sid].image_set
    if obj.novel:
        obj.familiarImages = familiarImages
        obj.novelImages = novelImagesG if obj.imageSet=='G' else novelImagesH
    else:
        obj.familiarImages = np.array(imageSetG) if obj.imageSet=='G' else np.array(imageSetH)
        obj.novelImages = None
    allSessions.append(obj)

hitThresh = 50
goodSessions = [obj for obj in allSessions if obj.hit.sum() > hitThresh]


# plots
firstStimDelay = np.concatenate([obj.firstStimDelay for obj in allSessions])
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


nHits = [obj.hit.sum() for obj in goodSessions]
hitRate = [obj.hitRate for obj in goodSessions]
falseAlarmRate = [obj.falseAlarmRate for obj in goodSessions]
dprime = [obj.dprime for obj in goodSessions]
abortRate = [obj.abortRate for obj in goodSessions]

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
    ax.set_title('r = '+str(round(r,2))+', p = '+pstr+', n = '+str(len(goodSessions)),fontsize=10)
    plt.tight_layout()


lickIntervals = np.concatenate([obj.lickIntervals for obj in goodSessions])
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


flashWithLickIntervals = np.concatenate([obj.flashWithLickIntervals for obj in goodSessions])
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

  
flashesToChange = np.concatenate([obj.flashesToChange[obj.changeTrials] for obj in goodSessions])    
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

flashesToAbort = np.concatenate([obj.flashesToLick[obj.abortedTrials] for obj in goodSessions])
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

flashesToOmitted = np.concatenate([obj.flashesToOmitted[~np.isnan(obj.flashesToOmitted)] for obj in goodSessions])
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
h,bins = np.histogram(flashesToOmitted,bins=np.arange(1,14))
ax.plot(bins[:-1],h/flashesToOmitted.size,'k')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlabel('flashes to omission')
ax.set_ylabel('fraction of trials with omitted flash')
plt.tight_layout()


hitLickLatencyByFlash = np.full((len(goodSessions),12),np.nan)
falseAlarmLickLatencyByFlash = hitLickLatencyByFlash.copy()
abortLickLatencyByFlash = hitLickLatencyByFlash.copy()
for sessionInd,obj in enumerate(goodSessions):
    for i in range(12):
        if i>3:
            hitLickLatencyByFlash[sessionInd,i] = np.nanmean(obj.flashLickLatency[obj.flashIsChange & (obj.flashTrialOutcome=='hit')][obj.flashesToChange[obj.hit]==i+1])
            falseAlarmLickLatencyByFlash[sessionInd,i] = np.nanmean(obj.flashLickLatency[obj.flashIsCatch & (obj.flashTrialOutcome=='false alarm')][obj.flashesToLick[obj.falseAlarm]==i+1])
        abortLickLatencyByFlash[sessionInd,i] = np.nanmean(obj.flashLickLatency[obj.flashIsAbort][obj.flashesToLick[obj.abortedTrials & ~obj.noLickAbortedTrials]==i+1])

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
x = np.arange(1,13)
for lat,clr,lbl in zip((hitLickLatencyByFlash,falseAlarmLickLatencyByFlash,abortLickLatencyByFlash),'kbr',('hit','false alarm','abort')):
    m = np.nanmean(lat,axis=0)
    s = np.nanstd(lat,axis=0)/(lat.shape[0]**0.5)
    ax.plot(x,m,color=clr,label=lbl)
    ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlabel('flash')
ax.set_ylabel('lick latency (s)')
ax.legend()
plt.tight_layout()


preTime = 1.5
postTime = 3.75
binSize = 1/60
bins = np.arange(-preTime,postTime+binSize/2,binSize)
lickPsth = []
for obj in goodSessions:
    lr = []
    for t in obj.flashStartTimes[obj.flashOmitted]:
        lt = obj.lickTimes[(obj.lickTimes>=t-preTime) & (obj.lickTimes<=t+postTime)] - t
        lr.append(np.histogram(lt,bins)[0] / binSize)
    lickPsth.append(np.mean(lr,axis=0))

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
x = bins[:-1]+binSize/2
m = np.mean(lickPsth,axis=0)
s = np.std(lickPsth,axis=0)/(len(lickPsth)**0.5)
ax.plot(x,m,'k')
ax.fill_between(x,m+s,m-s,color='k',alpha=0.25)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlim([-preTime,postTime])
ax.set_ylim([0,1.01*m.max()])
ax.set_xlabel('time from omitted flash (s)')
ax.set_ylabel('licks/s')
plt.tight_layout()

omittedFlashLickProb = np.stack([np.nanmean(obj.omittedFlashLickProb,axis=0) for obj in goodSessions])
postOmittedFlashLickProb = np.stack([np.nanmean(obj.postOmittedFlashLickProb,axis=0) for obj in goodSessions])
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
x = np.arange(1,13)
for lp,clr,lbl in zip((omittedFlashLickProb,postOmittedFlashLickProb),'kb',('omitted flash','post-omitted flash')):
    m = np.nanmean(lp,axis=0)
    s = np.nanstd(lp,axis=0)/(np.sum(~np.isnan(lp),axis=0)**0.5)
    ax.plot(x,m,color=clr,label=lbl)
    ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlabel('flashes to omission')
ax.set_ylabel('lick probability')
ax.legend()
plt.tight_layout()

omitLickLat = np.full((len(goodSessions),12),np.nan)
postOmitLickLat = omitLickLat.copy()
for sessionInd,obj in enumerate(goodSessions):
    for i in range(12):
        omitLickLat[sessionInd,i] = np.nanmean(obj.omittedFlashLickLat[obj.flashesToOmitted==i+1])
        postOmitLickLat[sessionInd,i] = np.nanmean(obj.postOmittedFlashLickLat[obj.flashesToOmitted==i+1])
        
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
x = np.arange(1,13)
for lat,clr,lbl in zip((omitLickLat,postOmitLickLat),'kb',('omitted flash','post-omitted flash')):
    m = np.nanmean(lat,axis=0)
    s = np.nanstd(lat,axis=0)/(np.sum(~np.isnan(lat),axis=0)**0.5)
    ax.plot(x,m,color=clr,label=lbl)
    ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlabel('flashes to omission')
ax.set_ylabel('lick latency (s)')
ax.legend()
plt.tight_layout()


flashLickProb = np.stack([np.nanmean(obj.flashLickProb,axis=0) for obj in goodSessions])
flashLickProbPrevTrialAborted = np.stack([np.nanmean(obj.flashLickProb[np.concatenate(([False],obj.abortedTrials[:-1]))],axis=0) for obj in goodSessions])
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
x = np.arange(1,13)
for lp,clr,lbl in zip((flashLickProb,flashLickProbPrevTrialAborted),'kb',('all trials','previous trial aborted')):
    m = np.mean(lp,axis=0)
    s = np.std(lp,axis=0)/(lp.shape[0]**0.5)
    ax.plot(x,m,color=clr,label=lbl)
    ax.fill_between(x,m+s,m-s,color=clr,alpha=0.25)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlabel('flashes (no change or omission)')
ax.set_ylabel('lick probability')
ax.legend()
plt.tight_layout()

flashLickProbTrials = np.stack([np.sum(~np.isnan(obj.flashLickProb),axis=0) for obj in goodSessions])
flashLickProbPrevTrialAbortedTrials = np.stack([np.sum(~np.isnan(obj.flashLickProb[np.concatenate(([False],obj.abortedTrials[:-1]))]),axis=0) for obj in goodSessions])
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


for selectedSessions,imageSet,novel,novelImgs in zip((novelHSessions,familiarGSessions,novelGSessions,familiarHSessions),'HGGH',(True,False,True,False),(novelImagesH,novelImagesH,novelImagesG,novelImagesG)):
    imageSetLabel = 'image set '+imageSet
    imageSetLabel += ' (novel)' if novel else ' (familiar)'
    imageSetLabel += ', n='+str(selectedSessions.sum())
    sessions = allGoodSessions[selectedSessions[goodSessions]]
    imgNames = np.array(imageSetG) if imageSet=='G' else np.array(imageSetH)
    bins = [None] + list(range(0,3600,600))
    
    imgRespRate = np.full((len(sessions),len(bins),len(imgNames),len(imgNames)),np.nan)
    imgHitRate = np.full((len(sessions),len(bins),len(imgNames)),np.nan)
    imgFalseAlarmRate = imgHitRate.copy()
    imgAbortRate = imgHitRate.copy()
    diag = np.eye(len(imgNames),dtype=bool)
    for sessionInd,obj in enumerate(sessions):
        for binInd,binStart in enumerate(bins):
            if binStart is None:
                binTrials = np.ones(obj.nTrials,dtype=bool)
            elif binInd==len(bins)-1:
                binTrials = obj.trialChangeTimes > binStart
            else:
                binTrials = (obj.trialChangeTimes >= binStart) & (obj.trialChangeTimes < bins[binInd+1])
            for i,initImg in enumerate(imgNames):
                imgTrials = binTrials & (obj.initialImage==initImg)
                imgAbortRate[sessionInd,binInd,i] = np.sum(imgTrials & obj.abortedTrials) / imgTrials.sum()
                for j,chImg in enumerate(imgNames):
                    trialInd = imgTrials & (obj.changeImage==chImg) & (obj.changeTrials | obj.catchTrials)
                    imgRespRate[sessionInd,binInd,i,j] = np.sum((obj.hit | obj.falseAlarm)[trialInd]) / trialInd.sum()
                r = imgRespRate[sessionInd,binInd].copy()
                imgFalseAlarmRate[sessionInd,binInd] = r[diag]
                r[diag] = np.nan
                imgHitRate[sessionInd,binInd] = np.nanmean(r,axis=0)  
    
    hitLickLat = []
    falseAlarmLickLat = []
    abortLickLat = []
    hitLickMeanLat = np.full((len(sessions),len(bins),len(imgNames)),np.nan)
    falseAlarmLickMeanLat = hitLickMeanLat.copy()
    abortLickMeanLat = hitLickMeanLat.copy()
    for sessionInd,obj in enumerate(sessions):
        for binInd,binStart in enumerate(bins):
            if binStart is None:
                binFlashes = np.ones(obj.nFlashes,dtype=bool)
            elif binInd==len(bins)-1:
                binFlashes = obj.flashStartTimes > binStart
            else:
                binFlashes = (obj.flashStartTimes >= binStart) & (obj.flashStartTimes < bins[binInd+1])
            for lat,meanLat,outcome,outcomeFlash in zip((hitLickLat,falseAlarmLickLat,abortLickLat),
                                                        (hitLickMeanLat,falseAlarmLickMeanLat,abortLickMeanLat),
                                                        ('hit','false alarm','abort'),
                                                        (obj.flashIsChange,obj.flashIsCatch,obj.flashIsAbort)):
                flashInd = binFlashes & (obj.flashTrialOutcome==outcome) & outcomeFlash
                lat.append(obj.flashLickLatency[flashInd])
                for i,img in enumerate(imgNames):
                    meanLat[sessionInd,binInd,i] = np.nanmean(obj.flashLickLatency[flashInd & (obj.flashImage==img)])
    
    if novel:
        imgOrder = np.argsort(np.nanmean(imgHitRate[:,0],axis=0))
    else:
        order = np.argsort(np.nanmean(imgHitRate[:,0],axis=0))
        imgOrder = np.concatenate((imgOrder[:2],order[~np.in1d(order,imgOrder[:2])]))
        
    fig = plt.figure()
    ax = plt.subplot(1,1,1)
    xticks = np.arange(len(imgNames))
    for r,clr,lbl in zip((hitLickMeanLat,falseAlarmLickMeanLat,abortLickMeanLat),'kbr',('hit','false alarm','abort')):
        ax.plot(xticks,np.nanmean(r[:,0],axis=0)[imgOrder],color=clr,label=lbl)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_ylim([0.2,0.5])
    ax.set_xticks(xticks)
    ax.set_xticklabels(imgNames[imgOrder])
    ax.set_ylabel('lick latency (s)')
    ax.set_title(imageSetLabel)
    ax.legend()
    plt.tight_layout()
    
    fig = plt.figure()
    ax = plt.subplot(1,1,1)
    xticks = np.arange(len(bins)-1)
    for r,clr,lbl in zip((hitLickMeanLat,falseAlarmLickMeanLat,abortLickMeanLat),'kbr',('hit','false alarm','abort')):
        ax.plot(xticks,np.nanmean(r[:,1:,imgOrder],axis=(0,2)),color=clr,label=lbl)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_ylim([0.2,0.5])
    ax.set_xticks(xticks)
    ax.set_xticklabels(bins[1:])
    ax.set_xlabel('start of time bin (s)')
    ax.set_ylabel('hit rate')
    ax.set_title(imageSetLabel)
    ax.legend()
    plt.tight_layout()

    
    fig = plt.figure()
    ax = plt.subplot(1,1,1)
    im = ax.imshow(np.nanmean(imgRespRate[:,0],axis=0)[imgOrder,:][:,imgOrder],cmap='magma')
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xticks(np.arange(len(imgNames)))
    ax.set_yticks(np.arange(len(imgNames)))
    ax.set_xlim([-0.5,len(imgNames)-0.5])
    ax.set_ylim([len(imgNames)-0.5,-0.5])
    ax.set_xticklabels(imgNames[imgOrder])
    ax.set_yticklabels(imgNames[imgOrder])
    ax.set_xlabel('change image')
    ax.set_ylabel('initial image')
    ax.set_title('response rate')
    cb = plt.colorbar(im,ax=ax,fraction=0.05,pad=0.04,shrink=0.5)
    plt.tight_layout()
    
    fig = plt.figure()
    ax = plt.subplot(1,1,1)
    xticks = np.arange(len(imgNames))
    for r,clr,lbl in zip((imgHitRate,imgFalseAlarmRate,imgAbortRate),'kbr',('hit rate','false alarm rate','abort rate')):
        ax.plot(xticks,np.nanmean(r[:,0],axis=0)[imgOrder],color=clr,label=lbl)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_ylim([0,1])
    ax.set_xticks(xticks)
    ax.set_xticklabels(imgNames[imgOrder])
    ax.set_ylabel('response rate')
    ax.legend()
    plt.tight_layout()
    
    fig = plt.figure()
    ax = plt.subplot(1,1,1)
    im = ax.imshow(np.nanmean(imgHitRate[:,1:],axis=0)[:,imgOrder],cmap='magma')
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xticks(np.arange(len(imgNames)))
    ax.set_yticks(np.arange(len(bins)-1))
    ax.set_xlim([-0.5,len(imgNames)-0.5])
    ax.set_ylim([len(bins)-1.5,-0.5])
    ax.set_xticklabels(imgNames[imgOrder])
    ax.set_yticklabels(bins[1:])
    ax.set_xlabel('change image')
    ax.set_ylabel('start of time bin (s)')
    ax.set_title('hit rate')
    cb = plt.colorbar(im,ax=ax,fraction=0.05,pad=0.04,shrink=0.5)
    plt.tight_layout()
    
    fig = plt.figure()
    ax = plt.subplot(1,1,1)
    xticks = np.arange(len(bins)-1)
    ax.plot(xticks,np.nanmean(imgHitRate[:,1:,imgOrder][:,:,2:],axis=(0,2)),color='r',label='novel')
    ax.plot(xticks,np.nanmean(imgHitRate[:,1:,imgOrder][:,:,:2],axis=(0,2)),color='k',label='familiar')
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_ylim([0,1])
    ax.set_xticks(xticks)
    ax.set_xticklabels(bins[1:])
    ax.set_xlabel('start of time bin (s)')
    ax.set_ylabel('hit rate')
    ax.legend()
    plt.tight_layout()
    

sessions = allGoodSessions[[obj.novel for obj in allGoodSessions]]
familiarHitRate = np.full((sessions.size,2000),np.nan)
novelHitRate = familiarHitRate.copy()
familiarAbortRate = familiarHitRate.copy()
novelAbortRate = familiarHitRate.copy()
binWidth = 30
bins = np.arange(0,3900,binWidth)
binCount = np.zeros_like(bins)
familiarHitRateBinned = np.zeros((sessions.size,bins.size))
novelHitRateBinned = familiarHitRateBinned.copy()
familiarAbortRateBinned = familiarHitRateBinned.copy()
novelAbortRateBinned = familiarHitRateBinned.copy()
for sessionInd,obj in enumerate(sessions):
    for imgs,(hr,ar),(hrb,arb) in zip((obj.familiarImages,obj.novelImages),((familiarHitRate,familiarAbortRate),(novelHitRate,novelAbortRate)),((familiarHitRateBinned,familiarAbortRateBinned),(novelHitRateBinned,novelAbortRateBinned))):   
        trialInd = obj.changeTrials & ~obj.autoRewarded & np.in1d(obj.changeImage,imgs)
        hr[sessionInd,:trialInd.sum()] = obj.hit[trialInd]
        binCount[:] = 0
        for i,r in zip(np.digitize(obj.trialChangeTimes[trialInd]-obj.trialStartTimes[0],bins),obj.hit[trialInd]):
            if i<bins.size:
                hrb[sessionInd,i] += r
                binCount[i] += 1
        hrb[sessionInd] /= binCount
        hrb[sessionInd][binCount==0] = np.nan
        
        trialInd = np.in1d(obj.initialImage,imgs)
        ar[sessionInd,:trialInd.sum()] = obj.abortedTrials[trialInd]
        binCount[:] = 0
        for i,r in zip(np.digitize(obj.trialStartTimes[trialInd]-obj.trialStartTimes[0],bins),obj.abortedTrials[trialInd]):
            if i<bins.size:
                arb[sessionInd,i] += r
                binCount[i] += 1
        arb[sessionInd] /= binCount
        arb[sessionInd][binCount==0] = np.nan

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for r,clr,lbl in zip((familiarHitRate,novelHitRate,familiarAbortRate,novelAbortRate),'rbmc',('hit rate familiar','hit rate novel','abort rate familiar','abort rate novel')):
    ax.plot(np.arange(r.shape[1])+1,np.nanmean(r,axis=0),clr,label=lbl)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlim([0,40])
ax.set_ylim([0,1])
ax.set_xlabel('trial')
ax.set_ylabel('response rate')
ax.legend()
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for r,clr,lbl in zip((familiarHitRateBinned,novelHitRateBinned,familiarAbortRateBinned,novelAbortRateBinned),'rbmc',('hit rate familiar','hit rate novel','abort rate familiar','abort rate novel')):
    ax.plot(bins+binWidth/2,np.nanmean(r,axis=0),clr,label=lbl)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlim([0,600])
ax.set_ylim([0,1])
ax.set_xlabel('time (s)')
ax.set_ylabel('response rate')
ax.legend()
plt.tight_layout()


sessions = allGoodSessions[[not obj.novel for obj in allGoodSessions]]
hitRate = np.full((sessions.size,2000),np.nan)
abortRate = familiarHitRate.copy()
hitRateBinned = np.zeros((sessions.size,bins.size))
abortRateBinned = hitRateBinned.copy()
for sessionInd,obj in enumerate(sessions):
    trialInd = obj.changeTrials & ~obj.autoRewarded
    hitRate[sessionInd,:trialInd.sum()] = obj.hit[trialInd]
    binCount[:] = 0
    for i,r in zip(np.digitize(obj.trialChangeTimes[trialInd]-obj.trialStartTimes[0],bins),obj.hit[trialInd]):
        if i<bins.size:
            hitRateBinned[sessionInd,i] += r
            binCount[i] += 1
    hitRateBinned[sessionInd] /= binCount
    hitRateBinned[sessionInd][binCount==0] = np.nan
    
    abortRate[sessionInd,:obj.nTrials] = obj.abortedTrials
    binCount[:] = 0
    for i,r in zip(np.digitize(obj.trialStartTimes-obj.trialStartTimes[0],bins),obj.abortedTrials):
        if i<bins.size:
            abortRateBinned[sessionInd,i] += r
            binCount[i] += 1
    abortRateBinned[sessionInd] /= binCount
    abortRateBinned[sessionInd][binCount==0] = np.nan
    
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for r,clr,lbl in zip((hitRate,abortRate),'rm',('hit rate','abort rate')):
    ax.plot(np.arange(r.shape[1])+1,np.nanmean(r,axis=0),clr,label=lbl)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlim([0,40])
ax.set_ylim([0,1])
ax.set_xlabel('trial')
ax.set_ylabel('response rate')
ax.legend()
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for r,clr,lbl in zip((hitRateBinned,abortRateBinned),'rm',('hit rate','abort rate')):
    ax.plot(bins+binWidth/2,np.nanmean(r,axis=0),clr,label=lbl)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlim([0,600])
ax.set_ylim([0,1])
ax.set_xlabel('time (s)')
ax.set_ylabel('response rate')
ax.legend()
plt.tight_layout()




        










# fit lick probability curve to exponential + gaussian
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

# or fit lick probability curve to exponential + gamma
x = np.arange(12)
y = np.mean(flashLickProb,axis=0)
bounds = ((0,-1e-10,0,0,-1e-10,-np.inf,0),(1,1e-10,12,1,1e-10,np.inf,12))
fitParams = fitCurve(expDecayPlusGamma,x,y,bounds=bounds)
ampExp,offsetExp,tau,ampGamma,offsetGamma,shape,loc = fitParams
fity = expDecayPlusGamma(x,*fitParams)

impulsivityLickProb = expDecay(x,ampExp,offsetExp,tau)
timingLickProb = gamma(x,ampGamma,offsetGamma,shape,loc)

impulsivityLickProbNorm = impulsivityLickProb / impulsivityLickProb.max()
timingLickProbNorm = timingLickProb / timingLickProb.max()

# plot fit
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
x = np.arange(1,13)
ax.plot(x,np.mean(flashLickProb,axis=0),'k',label='all trials')
ax.plot(x,np.mean(flashLickProbPrevTrialAborted,axis=0),'g',label='previous trial aborted')
ax.plot(x,fity,'r',label='fit')
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
sessions = allGoodSessions[novelHSessions[goodSessions]]
imageSet = 'H'
novel = True

# exclude flashes after trial outcome decided and ignore autorewarded trials
regressors = ['impulsivity','timing','post-omission','change','novelty']
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
    postOmittedInd = np.where(obj.flashOmitted)[0] + 1
    postOmission = np.zeros(obj.nFlashes)
    postOmission[postOmittedInd[postOmittedInd<postOmission.size]]=True
    x['post-omission'].append(postOmission)
    x['change'].append(obj.flashIsChange.astype(float))
    novelImgs = (novelImagesG if imageSet=='G' else novelImagesH) if novel else None
    x['novelty'].append(np.in1d(obj.flashImage,novelImgs).astype(float))
    y.append(flashHasLick)
        
excludedFlashes = [np.isnan(a) for a in y]


# fit individual sessions
hyperparams = []
evidence = []
modelWeights = []
hessian = []
cvLikelihood = []
cvProbNoLick = []
accuracy = []
accuracyChange = []
accuracyOmitted = []
for i in range(len(sessions)):
    print(i)
    d = {'inputs': {key: val[i][~excludedFlashes[i]][:,None] for key,val in x.items() if key not in (None,)},
         'y': y[i][~excludedFlashes[i]]}
    d['inputs']['bias'] = np.ones(np.sum(~excludedFlashes[i]))

    weights = {key: 1 for key in d['inputs']}

    nWeights = sum(weights.values())

    hyper= {'sigInit': 2**4.,
            'sigma': [2**-4.] * nWeights}

    optList = ['sigma']

    t = time.perf_counter()
    hyp, evd, wMode, hess_info = psytrack.hyperOpt(d, hyper, weights, optList)
    print(time.perf_counter()-t)
    
    hyperparams.append(hyp)
    evidence.append(evd)
    modelWeights.append(wMode)
    hessian.append(hess_info)

    cvFolds = 5
    cvTrials = d['y'].size - (d['y'].size % cvFolds)
    t = time.perf_counter()
    likelihood,probNoLick = psytrack.crossValidate(psytrack.trim(d,END=cvTrials), hyper, weights, optList, F=cvFolds, seed=0)
    print(time.perf_counter()-t)
    
    cvLikelihood.append(likelihood)
    cvProbNoLick.append(probNoLick)
    
    lick = d['y'][:cvTrials]==2 if 2 in d['y'][:cvTrials] else d['y'][cvTrials]==1
    accuracy.append(np.abs(lick - probNoLick))
    accuracyChange.append(accuracy[-1][d['inputs']['change'].astype(bool).flatten()[:cvTrials]])
    accuracyOmitted.append(accuracy[-1][d['inputs']['omission'].astype(bool).flatten()[:cvTrials]])


flashStartTimes = [obj.flashStartTimes[~excludedFlashes[i]]-obj.flashStartTimes[0] for i,obj in enumerate(sessions)]

fig = plt.figure(figsize=(8,8)) 
ylim = [min(0,1.05*min([w.min() for w in modelWeights])),1.05*max([w.max() for w in modelWeights])]
for i,lbl in enumerate(sorted(weights.keys())):
    ax = fig.add_subplot(nWeights,1,i+1)
    for j,w in enumerate(modelWeights):
        ax.plot(flashStartTimes[j],w[i],'k',alpha=0.25)
    # ax.plot(np.mean(np.stack([w[i][:2500] for w in modelWeights]),axis=0),'r',lw=2)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False,labelsize=10)
    ax.set_xlabel('time (s)')
    ax.set_title(sorted(regressors+['bias'])[i])
plt.tight_layout()


# # fit all sessions
# d = {'inputs': {key: np.concatenate(val)[~np.concatenate(excludedFlashes)][:,None] for key,val in x.items()},
#      'y': np.concatenate(y)[~np.concatenate(excludedFlashes)],
#      'dayLength': np.array([np.sum(~b) for b in excludedFlashes])}

# weights = {key: 1 for key in d['inputs']}

# nWeights = sum(weights.values())

# hyper= {'sigInit': 2**4.,
#         'sigma': [2**-4.] * nWeights,
#         'sigDay': [2**-4.] * nWeights}

# optList = ['sigma','sigDay']

# t = time.perf_counter()
# hyp, evd, wMode, hess_info = psytrack.hyperOpt(d, hyper, weights, optList)
# print(time.perf_counter()-t)

# cvFolds = 5
# cvTrials = d['y'].size - (d['y'].size % cvFolds)
# t = time.perf_counter()
# cvLikelihood,cvProbNoLick = psytrack.crossValidate(psytrack.trim(d,END=cvTrials), hyper, weights, optList, F=cvFolds, seed=0)
# print(time.perf_counter()-t)
# cvProbLick = 1-cvProbNoLick

# yModel = (cvProbLick>=0.5).astype(float)
# accuracy = 1 - (np.abs(y[:cvTrials]-yModel).sum() / cvTrials)

# fig = plt.figure(figsize=(8,8))
# ylim = [min(0,1.05*wMode.min()),1.05*wMode.max()]
# for wi,(w,lbl) in enumerate(zip(wMode,sorted(weights.keys()))):
#     ax = fig.add_subplot(nWeights,1,wi+1)
#     sessionFlashes = d['dayLength']
#     sessionStartStop = np.concatenate(([0],np.cumsum(sessionFlashes)))
#     for si in range(len(sessions)):
#         ax.plot(np.arange(sessionFlashes[si])+1,w[sessionStartStop[si]:sessionStartStop[si+1]],'k',alpha=0.5)
#     for side in ('right','top'):
#         ax.spines[side].set_visible(False)
#     ax.tick_params(direction='out',top=False,right=False,labelsize=10)
#     ax.set_ylim(ylim)
# plt.tight_layout()
















