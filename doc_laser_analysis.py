# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 11:26:01 2019

@author: svc_ccg
"""

from __future__ import division
import os
import re
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import fileIO
from sync import sync
import probeSync



class DocLaser():
    
    def __init__(self,pklPath):
        
        self.laserPowerDict = {
                               0: {0.26: 0.5, 0.43: 1, 0.6: 1.5, 0.77: 2, 0.93: 2.5,
                                   0.81: 0.5, 1.79: 1, 2.76: 1.5, 3.74: 2, 4.71: 2.5,
                                   5: 2.5},
                               1: {0.32: 0.5, 0.52: 1, 0.72: 1.5, 0.92: 2, 1.11: 2.5,
                                   0.38: 0.5, 0.6: 1, 0.83: 1.5, 1.05: 2, 1.28: 2.5,
                                   0.4: 0.5, 0.64: 1, 0.76: 1.25, 0.88: 1.5, 1.12: 2, 1.36: 2.5,
                                   1.38: 2.5, 1.5: 2.5}
                              }
        
        pkl = pd.read_pickle(pklPath)
        self.pklPath = pklPath
        self.expDate = str(pkl['start_time'].date())
    
        self.params = pkl['items']['behavior']['params']
        if self.params['periodic_flash'] is not None:
            flashDur,grayDur = self.params['periodic_flash']
            flashInterval = flashDur + grayDur
        
        self.frameRate = 60
        self.monitorLag = self.params['laser_params']['monitor_lag']
        
        trialLog = pkl['items']['behavior']['trial_log']
        laserLog = pkl['items']['behavior']['laser_trials']
#        changeLog = pkl['items']['behavior']['stimuli']['images']['change_log']
        
        ntrials = len(trialLog)
        self.trialStartTimes = np.full(ntrials,np.nan)
        self.trialStartFrames = np.full(ntrials,np.nan)
        self.trialEndTimes = np.full(len(trialLog),np.nan)
        self.abortedTrials = np.zeros(len(trialLog),dtype=bool)
        self.abortTimes = np.full(len(trialLog),np.nan)
        self.abortFrames = np.full(len(trialLog),np.nan)
        self.scheduledChangeTimes = np.full(len(trialLog),np.nan)
        self.changeTimes = np.full(len(trialLog),np.nan)
        self.changeFrames = np.full(len(trialLog),np.nan)
        self.changeTrials = np.zeros(len(trialLog),dtype=bool)
        self.catchTrials = np.zeros(len(trialLog),dtype=bool)
        self.preChangeImage = ['' for _ in range(len(trialLog))]
        self.changeImage = ['' for _ in range(len(trialLog))]
        self.rewardTimes = np.full(len(trialLog),np.nan)
        self.autoReward = np.zeros(len(trialLog),dtype=bool)
        self.hit = np.zeros(len(trialLog),dtype=bool)
        self.miss = np.zeros(len(trialLog),dtype=bool)
        self.falseAlarm = np.zeros(len(trialLog),dtype=bool)
        self.correctReject = np.zeros(len(trialLog),dtype=bool)
        for i,trial in enumerate(trialLog):
            events = [event[0] for event in trial['events']]
            for event,epoch,t,frame in trial['events']:
                if event=='trial_start':
                    self.trialStartTimes[i] = t
                    self.trialStartFrames[i] = frame
                elif event=='trial_end':
                    self.trialEndTimes[i] = t
                elif event=='stimulus_window' and epoch=='enter':
                    ct = trial['trial_params']['change_time']
                    if self.params['periodic_flash'] is not None:
                        ct *= flashInterval
                        ct -= self.params['pre_change_time']-flashInterval
                    self.scheduledChangeTimes[i] = t + ct
                elif event=='abort':
                    self.abortedTrials[i] = True
                    self.abortTimes[i] = t
                    self.abortFrames[i] = frame
                elif event in ('stimulus_changed','sham_change'):
                    self.changeTimes[i] = t
                    self.changeFrames[i] = frame
                elif event=='hit':
                    self.hit[i] = True
                elif event=='miss':
                    self.miss[i] = True 
                elif event=='false_alarm' and 'rejection' not in events:
                    self.falseAlarm[i] = True
                elif event=='rejection':
                    self.correctReject[i] = True
            if not self.abortedTrials[i]:
                if trial['trial_params']['catch']:
                    self.catchTrials[i] = True
                else:
                    if len(trial['stimulus_changes'])>0:
                        self.changeTrials[i] = True
                        self.preChangeImage[i] = trial['stimulus_changes'][0][0][0]
                        self.changeImage[i] = trial['stimulus_changes'][0][1][0]
            if len(trial['rewards']) > 0:
                self.rewardTimes[i] = trial['rewards'][0][1]
                self.autoReward[i] = trial['trial_params']['auto_reward']
                
        self.laserAmp = np.full(len(trialLog),np.nan)
        self.laserFlashOffset = np.full(len(trialLog),np.nan)
        self.laserFrameOffset = np.full(len(trialLog),np.nan)
        self.laserOnFrame = np.full(len(trialLog),np.nan)
        self.laserOnBeforeAbort = np.zeros(len(trialLog),dtype=bool)
        expectedLaserOffsets = [int(x[0]*flashInterval*self.frameRate+x[1]+self.monitorLag) for x in self.params['laser_params']['offset']]
        for laserTrial in laserLog:
            if 'actual_laser_frame' in laserTrial:
                i = laserTrial['trial']
                self.laserOnFrame[i] = laserTrial['actual_laser_frame']
                if 'laser' in laserTrial:
                    if 'amp' in laserTrial:
                        amp = laserTrial['amp']
                        self.laserAmp[i] = amp
                if 'actual_change_frame' in laserTrial:
                    laserOffset = laserTrial['actual_laser_frame']-laserTrial['actual_change_frame']
                    if laserOffset in expectedLaserOffsets:
                        self.laserFrameOffset[i] = laserOffset
                        self.laserFlashOffset[i] = laserTrial['actual_laser_flash']-laserTrial['actual_change_flash']
                else:
                    self.laserOnBeforeAbort[i] = True
                    
        outcomeTimes = np.zeros(len(trialLog))
        outcomeTimes[self.abortedTrials] = self.abortTimes[self.abortedTrials]
        outcomeTimes[~self.abortedTrials] = self.changeTimes[~self.abortedTrials]
        self.engaged = np.array([np.sum(self.hit[(outcomeTimes>t-60) & (outcomeTimes<t+60)]) > 1 for t in outcomeTimes])
                
        frameIntervals = pkl['items']['behavior']['intervalsms']/1000
        frameTimes = np.concatenate(([0],np.cumsum(frameIntervals)))
        frameTimes += self.trialStartTimes[0] - frameTimes[int(self.trialStartFrames[0])]
        
        self.lickFrames = pkl['items']['behavior']['lick_sensors'][0]['lick_events']
        self.lickTimes = frameTimes[self.lickFrames]
        
        self.omitFlashProb = pkl['items']['behavior']['stimuli']['images']['flash_omit_probability']
        self.omitFlashFrames = pkl['items']['behavior']['stimuli']['images']['flashes_omitted']
        self.omitFlashTimes = frameTimes[self.omitFlashFrames]
        self.postOmitLick = np.zeros(len(self.omitFlashFrames),dtype=bool)
        for i,ft in enumerate(self.omitFlashTimes):
            t = ft+flashInterval
            if any((self.lickTimes > t+self.params['response_window'][0]) & (self.lickTimes < t+self.params['response_window'][1])):
                self.postOmitLick[i] = True
        
        self.postChangeHit = np.zeros(len(trialLog),dtype=bool)
        self.postChangeMiss = np.zeros(len(trialLog),dtype=bool)
        self.postChangeFalseAlarm = np.zeros(len(trialLog),dtype=bool)
        self.postChangeCorrectReject = np.zeros(len(trialLog),dtype=bool)
        for i in range(ntrials):
            if self.miss[i] or self.correctReject[i]:
                t = self.changeTimes[i]+flashInterval
                if any((self.lickTimes > t+self.params['response_window'][0]) & (self.lickTimes < t+self.params['response_window'][1])):
                    if self.miss[i]:
                        self.postChangeHit[i] = True
                    else:
                        self.postChangeFalseAlarm[i] = True
                else:
                    if self.miss[i]:
                        self.postChangeMiss[i] = True
                    else:
                        self.postChangeCorrectReject[i] = True
        
        

def getLickLatency(lickTimes,eventTimes,offset=0):
    firstLickInd = np.searchsorted(lickTimes,eventTimes+offset)
    noLicksAfter = firstLickInd==lickTimes.size
    firstLickInd[noLicksAfter] = lickTimes.size-1
    lickLat = lickTimes[firstLickInd]-eventTimes
    lickLat[noLicksAfter] = np.nan
    return lickLat


def plotPerformance(exps,label=None,sessions=None,led=None,showReactionTimes=False):
    label = '' if label is None else label+' '
    frameRate = exps[0].frameRate
    monitorLag = exps[0].monitorLag
    respWin = exps[0].params['response_window']
    
    if sessions is None:
        sessions = list(range(len(exps)))
    laserFrameOffset = np.concatenate([exps[i].laserFrameOffset for i in sessions])
    laserAmp = np.concatenate([exps[i].laserAmp for i in sessions])
    changeTrials = np.concatenate([exps[i].changeTrials for i in sessions])
    catchTrials = np.concatenate([exps[i].catchTrials for i in sessions])
    hit = np.concatenate([exps[i].hit for i in sessions])
    miss = np.concatenate([exps[i].miss for i in sessions])
    falseAlarm = np.concatenate([exps[i].falseAlarm for i in sessions])
    correctReject = np.concatenate([exps[i].correctReject for i in sessions])
    postChangeHit = np.concatenate([exps[i].postChangeHit for i in sessions])
    postChangeFalseAlarm = np.concatenate([exps[i].postChangeFalseAlarm for i in sessions])
    postOmitLick = np.concatenate([exps[i].postOmitLick for i in sessions])
    engaged = np.concatenate([exps[i].engaged for i in sessions])
    
    hitRate = []
    falseAlarmRate = []
    offsets = np.concatenate((np.unique(laserFrameOffset[~np.isnan(laserFrameOffset)]),[np.nan]))
    if np.sum(~np.isnan(offsets))>1:
        xdata = laserFrameOffset
        xvals = offsets
        xticks = (offsets-monitorLag)/frameRate*1000
        xticks[-1] = xticks[-2]+(xticks[-2]-xticks[-3])
        xticklabels = [int(i) for i in xticks[:-1]]+['no opto']
        xlabel = 'Laser onset relative to change (ms)'
    else:
        amps = np.concatenate(([np.nan],np.unique(laserAmp[~np.isnan(laserAmp)])))
        xdata = laserAmp
        xvals = amps
        xticks = list(amps)
        xticks[0] = 0
        xticklabels = ['no opto']+xticks[1:]
        xlabel = 'Laser amp (V)'
    fig = plt.figure(figsize=(6,8))
    for a,(trialTypes,resps,flashLbl) in enumerate(zip(((changeTrials,catchTrials),(miss,correctReject)),((hit,falseAlarm),(postChangeHit,postChangeFalseAlarm)),('change flash','post-change flash'))):
        ax = fig.add_subplot(2,1,a+1)
        if a==1 and len(postOmitLick)>0:
            omitLickProb = postOmitLick.sum()/postOmitLick.size
            ax.plot([xticks[0],xticks[-1]],[omitLickProb]*2,'--',color='0.5')
        for trials,resp,clr,lbl,txty in zip(trialTypes,resps,'kr',('change','catch'),(1.07,1.02)):
            r = []
            for j,(xval,x) in enumerate(zip(xvals,xticks)):
                xTrials = np.isnan(xdata) if np.isnan(xval) else xdata==xval
                i = trials & xTrials
                ntotal = i.sum()
                i = i & engaged
                n = i.sum()
                r.append(resp[i].sum()/n)
                fig.text(x,txty,str(n)+'/'+str(ntotal),color=clr,transform=ax.transData,va='bottom',ha='center',fontsize=8)
            ax.plot(xticks,r,'o-',color=clr,label=lbl)
            if a==0:
                if lbl=='change':
                    hitRate.append(r)
                else:
                    falseAlarmRate.append(r)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_ylim([0,1.02])
        ax.set_xticks(xticks)
        if a==1:
            ax.set_xticklabels(xticklabels)
            ax.set_xlabel(xlabel)
        else:
            ax.set_xticklabels([])
        ax.set_ylabel('Response rate ('+flashLbl+')')
        ax.legend()
    
    if showReactionTimes:
        lickLatency = np.concatenate([getLickLatency(obj.lickTimes,obj.changeTimes,respWin[0]) for i,obj in enumerate(exps) if i in sessions])
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        for w in respWin:
            ax.plot([0,xticks[-1]],[w*1000]*2,'--',color='0.75')
        for trials,resp,clr,lbl in zip((changeTrials,catchTrials),(hit,falseAlarm),'kr',('hit','false alarm')):
            r = []
            for xval,x in zip(xvals,xticks):
                xTrials = np.isnan(xdata) if np.isnan(xval) else xdata==xval
                i = trials & resp & xTrials & engaged
                r.append(1000*lickLatency[i])
                ax.plot(x+np.zeros(len(r[-1])),r[-1],'o',mec=clr,mfc='none')
            ax.plot(xticks,[np.nanmean(y) for y in r],'o',mec=clr,mfc=clr,label=lbl)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_ylim([0,900])
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticklabels)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Reaction time (ms)')
        ax.legend()
            
    return hitRate,falseAlarmRate
            
            
def plotLicks(obj,preTime=1.5,postTime=0.75):
    offsets = np.concatenate((np.unique(obj.laserFrameOffset[~np.isnan(obj.laserFrameOffset)]),[np.nan]))
    for offset in offsets:
        fig = plt.figure()
        offsetTrials = np.isnan(obj.laserFrameOffset) if np.isnan(offset) else obj.laserFrameOffset==offset
        for i,(trialType,lbl) in enumerate(zip((obj.changeTrials,obj.catchTrials),('change','catch'))):
            ax = fig.add_subplot(2,1,i+1)
            for n,ct in enumerate(obj.changeTimes[trialType & obj.engaged & offsetTrials]):
                licks = obj.lickTimes[(obj.lickTimes>ct-preTime) & (obj.lickTimes<ct+postTime)]-ct
                ax.vlines(licks,n-0.5,n+0.5,colors='k')
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False)
            ax.set_xlim([-preTime,postTime])
            ax.set_ylim([-1,n+1])
            ax.set_xlabel('Time relative to '+lbl+' (s)')   
            ax.set_ylabel('Trial')
            if i==0:
                ax.set_title('offset '+str((offset-obj.monitorLag)/obj.frameRate*1000)+' s')
        plt.tight_layout()



pklFiles = []
while True:
    f = fileIO.getFile('choose pkl file',fileType='*.pkl')
    if f!='':
        pklFiles.append(f)
    else:
        break

if len(pklFiles)>0:
    exps = []
    for f in pklFiles:
        obj = DocLaser(f)
        exps.append(obj)
        
        
hitRate = []
falseAlarmRate = []
for obj in exps:
    h,fa = plotPerformance([obj],label=obj.expDate)
    hitRate.append(h)
    falseAlarmRate.append(fa)
    
    plotLicks(obj,preTime=1.5,postTime=1.5)
    

plotPerformance(exps)

    
#f = fileIO.getFile('choose experiments file',fileType='*.xlsx')
#exps = pd.read_excel(f)
#mouseDir = os.path.dirname(f)
#experimentLabel = 'region/power'
#pklFiles = []
#led0Regions = []
#led1Regions = []
#for i,row in exps.iterrows():
#    if row['experiment']==experimentLabel:
#        pklFiles.append(glob.glob(os.path.join(mouseDir,re.sub('[.]','',row['date']),'*.pkl'))[0])
#        led0Regions.append(row['led 0'])
#        led1Regions.append(row['led 1'])


#regions = np.unique([r for r in led0Regions+led1Regions if str(r)!='nan'])
#for reg in regions:
#    sessions,led = zip(*[(i,j) for j,ledReg in enumerate((led0Regions,led1Regions)) for i,r in enumerate(ledReg) if r==reg])
#    plotPerformance(exps,label=reg,sessions=sessions,led=led)



#
syncFiles = []
while True:
    f = fileIO.getFile('choose sync file',fileType='*.h5')
    if f!='':
        syncFiles.append(f)
    else:
        break

ledOnset = []
for i,(f,obj) in enumerate(zip(syncFiles,exps)):
    syncDataset = sync.Dataset(f)
    
    frameRising, frameFalling = probeSync.get_sync_line_data(syncDataset, 'vsync_stim')
    vsyncTimes = frameFalling[1:] if frameFalling[0] < frameRising[0] else frameFalling
    frameAppearTimes = vsyncTimes + obj.monitorLag/obj.frameRate
    
    binWidth = 0.001
    for laserInd,ch in enumerate((11,1)):
        laserRising,laserFalling = probeSync.get_sync_line_data(syncDataset,channel=ch)
        if len(laserRising)>0:
            laserTrials = obj.laser==laserInd
            ct = frameAppearTimes[obj.changeFrames[laserTrials & (obj.changeTrials | obj.catchTrials)].astype(int)]
            fig = plt.figure(figsize=(6,6))
            for j,(t,xlbl) in enumerate(zip((laserRising,laserFalling),('onset','offset'))):
                timeFromChange = t[~obj.laserOnBeforeAbort[laserTrials]]-ct
                if xlbl=='onset':
                    ledOnset.append(timeFromChange)
                ax = fig.add_subplot(2,1,j+1)
                ax.hist(1000*timeFromChange,bins=1000*np.arange(round(min(timeFromChange),3)-binWidth,round(max(timeFromChange),3)+binWidth,binWidth),color='k')
                for side in ('right','top'):
                    ax.spines[side].set_visible(False)
                ax.tick_params(direction='out',top=False,right=False)
                ax.set_xlabel('Time from change on monitor to led '+xlbl+' (ms)')
                ax.set_ylabel('Count')
                if j==0:
                    ax.set_title('laser '+str(laserInd))
            plt.tight_layout()
            
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
onset = np.concatenate(ledOnset)
ax.hist(1000*onset,bins=1000*np.arange(round(min(onset),3)-binWidth,round(max(onset),3)+binWidth,binWidth),color='k')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlabel('Time from change on monitor to led onset (ms)')
ax.set_ylabel('Count')
plt.tight_layout()



        