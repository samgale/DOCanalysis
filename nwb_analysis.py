# -*- coding: utf-8 -*-
"""
Created on Thu May 12 10:51:40 2022

@author: samg
"""

import gc
import os
import glob
import numpy as np
import scipy.signal
import scipy.ndimage
import matplotlib
matplotlib.use('agg')
# matplotlib.use('tkagg')
import matplotlib.pyplot as plt
from pynwb import NWBHDF5IO
from allensdk.brain_observatory.ecephys.behavior_ecephys_session import (
    BehaviorEcephysSession)


nwb_base = r"\\allen\programs\mindscope\workgroups\np-behavior\vbn_data_release\vbn_s3_cache\visual-behavior-neuropixels-0.1.0\ecephys_sessions"
nwb_paths = glob.glob(os.path.join(nwb_base, '*nwb'))

outputPath = r'\\allen\programs\mindscope\workgroups\np-exp\VBN_NWB_validation'



def getSdf(spikes,startTimes,windowDur,sampInt=0.001,filt='exponential',filtWidth=0.005,avg=True):
        t = np.arange(0,windowDur+sampInt,sampInt)
        counts = np.zeros((startTimes.size,t.size-1))
        for i,start in enumerate(startTimes):
            counts[i] = np.histogram(spikes[(spikes>=start) & (spikes<=start+windowDur)]-start,t)[0]
        if filt in ('exp','exponential'):
            filtPts = int(5*filtWidth/sampInt)
            expFilt = np.zeros(filtPts*2)
            expFilt[-filtPts:] = scipy.signal.exponential(filtPts,center=0,tau=filtWidth/sampInt,sym=False)
            expFilt /= expFilt.sum()
            sdf = scipy.ndimage.convolve1d(counts,expFilt,axis=1)
        else:
            sdf = scipy.ndimage.gaussian_filter1d(counts,filtWidth/sampInt,axis=1)
        if avg:
            sdf = sdf.mean(axis=0)
        sdf /= sampInt
        return sdf[:-1],t[:-2]
    

def makeLickPlots(session):
    
    sessionID = session.metadata['ecephys_session_id']
    trials = session.trials
    stim = session.stimulus_presentations
    licks = session.licks
    
    fig = plt.figure(figsize=(8,8))
    gs = matplotlib.gridspec.GridSpec(9,2)
    preTime = 0.75
    postTime = 3
    binSize = 0.05
    bins = np.arange(-preTime,postTime+binSize/2,binSize)
    for trialType,i,j in zip(('hit','miss','false_alarm','correct_reject','auto_rewarded','omitted'),(0,3,6)*2,(0,1)*3):
        ax = fig.add_subplot(gs[i:i+2,j])
        if trialType == 'omitted':
            t = stim['start_time'][stim['active'] & stim['omitted']]
        elif trialType == 'auto_rewarded':
            t = trials['change_time_no_display_delay'][trials[trialType]]
        else:
            t = trials['change_time_no_display_delay'][trials[trialType] & ~trials['auto_rewarded']]
        lickRaster = []
        for tr,st in enumerate(t):
            lt = licks['timestamps'] - st
            lickRaster.append(lt[(lt >= -preTime) & (lt <= postTime)])
            ax.vlines(lickRaster[-1],tr+0.5,tr+1.5,colors='k')       
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_xlim([-preTime,postTime])
        ax.set_ylim([0.5,t.size+0.5])
        ax.set_yticks([1,t.size])
        if j==0:
            ax.set_ylabel('trial')
        ax.set_title(trialType)
        
        if len(lickRaster) > 0:
            ax = fig.add_subplot(gs[i+2,j])
            lickPsth = np.mean([np.histogram(r,bins)[0] for r in lickRaster],axis=0) / binSize
            ax.plot(bins[:-1]+binSize/2,lickPsth,color='k')
            for side in ('right','top'):
                ax.spines[side].set_visible(False)
            ax.tick_params(direction='out',top=False,right=False)
            ax.set_xlim([-preTime,postTime])
            ax.set_ylim([0,1.01*lickPsth.max()])
            if i==6:
                ax.set_xlabel('time from change/catch/omission (s)')
            if j==0:
                ax.set_ylabel('licks/s')
    plt.tight_layout()
    plt.savefig(os.path.join(outputPath,'licks','lick_raster',str(sessionID)+'_lick_raster.png'))
    plt.close(fig)

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    lickTimes = np.array(licks['timestamps'])
    for trialType,clr in zip(('hit','miss','false_alarm','correct_reject'),'grcm'):
        changeTimes = np.array(trials['change_time_no_display_delay'][trials[trialType] & ~trials['auto_rewarded']])
        firstLickInd = np.searchsorted(lickTimes,changeTimes)
        hasLick = firstLickInd < lickTimes.size
        firstLickLat = lickTimes[firstLickInd[hasLick]]-changeTimes[hasLick]
        sortLat = np.sort(firstLickLat)
        cumProb = [np.sum(firstLickLat<=i)/firstLickLat.size for i in sortLat]
        ax.plot(sortLat,cumProb,color=clr,label=trialType)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xlim([0,1.5])
    ax.set_ylim([0,1.02])
    ax.set_xlabel('first lick latency (s)')
    ax.set_ylabel('cumulative prob.')
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outputPath,'licks','first_lick',str(sessionID)+'_first_lick.png'))
    plt.close(fig)       

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    for img in np.unique(trials['change_image_name']):
        tr = (trials['change_image_name']==img) & ~trials['auto_rewarded']
        hitTrials = tr & trials['hit']
        h = np.sum(hitTrials)
        m = np.sum(tr & trials['miss'])
        hr = h/(h+m)
        rt = np.median(trials['reward_time'][hitTrials] - trials['change_time_no_display_delay'][hitTrials])
        ax.plot(hr,rt,'o',mec='k',mfc='none')
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_xlim([0,1])
        ax.set_xlabel('hit rate (each image)')
        ax.set_ylabel('reaction time (s)')
    plt.tight_layout()
    plt.savefig(os.path.join(outputPath,'licks','reaction_time',str(sessionID)+'_reaction_time.png'))
    plt.close(fig)


def makeSaccadePlots(session):
    
    sessionID = session.metadata['ecephys_session_id']
    eyeTracking =session.eye_tracking
    
    t = np.array(eyeTracking['timestamps'])
    pupilX = eyeTracking['pupil_center_x'] - eyeTracking['cr_center_x']
    pupilX[eyeTracking['likely_blink']] = np.nan
    
    saccadeSmoothPts = 3
    saccadeThresh = 5
    saccadeRefractoryPeriod = 0.1
    
    n = saccadeSmoothPts//2
    vel = np.diff(pupilX)/np.diff(t)
    velSmoothed = np.convolve(vel,np.ones(saccadeSmoothPts)/saccadeSmoothPts,mode='same')
    velSmoothed[:n] = vel[:n].mean()
    velSmoothed[-n:] = vel[-n:].mean()
    
    # find peaks in pupil velocity
    v = velSmoothed
    thresh = saccadeThresh * np.nanstd(v)
    negSaccades = np.where((v < -thresh) & np.concatenate(([False],v[1:] < v[:-1])) & np.concatenate((v[:-1] < v[1:],[False])))[0]
    posSaccades = np.where((v > thresh) & np.concatenate(([False],v[1:] > v[:-1])) & np.concatenate((v[:-1] > v[1:],[False])))[0]
    
    # remove peaks that are too close in time
    negSaccades = negSaccades[np.concatenate(([True],np.diff(t[negSaccades]) > saccadeRefractoryPeriod))]
    posSaccades = posSaccades[np.concatenate(([True],np.diff(t[posSaccades]) > saccadeRefractoryPeriod))]
    
    # remove negative peaks too closely following positive peaks and vice versa
    peakTimeDiff = t[negSaccades]-t[posSaccades][:,None]
    negSaccades = negSaccades[np.all(np.logical_or(peakTimeDiff<0,peakTimeDiff > saccadeRefractoryPeriod),axis=0)]
    posSaccades = posSaccades[np.all(np.logical_or(peakTimeDiff>0,peakTimeDiff < -saccadeRefractoryPeriod),axis=1)]
    
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.plot(t,pupilX,'k')
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('pupil X position')
    plt.tight_layout()
    plt.savefig(os.path.join(outputPath,'saccades','xpos',str(sessionID)+'_xpos.png'))
    plt.close(fig)
    
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    firstSaccade = min(negSaccades[0],posSaccades[0])
    tind = np.where((t>t[firstSaccade]-1) & (t<t[firstSaccade]+60))[0]
    ax.plot(t[tind],pupilX[tind],'k')
    for saccades,clr in zip((negSaccades,posSaccades),'rb'):
        sind = saccades[(saccades>=tind[0]) & (saccades<=tind[-1])]
        ax.plot(t[sind],pupilX[sind],'o',mec=clr,mfc=clr)
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('pupil X position')
    plt.tight_layout()
    plt.savefig(os.path.join(outputPath,'saccades','saccade_example',str(sessionID)+'_saccade_example.png'))
    plt.close(fig)
    
    
    fig = plt.figure()
    preTime = 0.1
    postTime = 0.2
    frameRate = 60
    eyePlotTime = np.arange(-preTime,postTime+1/frameRate,1/frameRate)
    for ind,(saccades,clr,lbl) in enumerate(zip((negSaccades,posSaccades),'rb',('negative saccades','positive saccades'))):
        ax = fig.add_subplot(2,1,ind+1)
        px = []
        for st in t[saccades]:
            if st >= preTime and st+postTime <= t[-1]:
                i = (t >= st-preTime) & (t <= st+postTime)
                px.append(np.interp(eyePlotTime,t[i]-st,pupilX[i]))
                px[-1] -= np.nanmean(px[-1][eyePlotTime<-0.05])
                ax.plot(eyePlotTime,px[-1],color=clr,alpha=0.005)
        pxMean = np.nanmean(px,axis=0)
        ax.plot(eyePlotTime,pxMean,color=clr,lw=2)
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_xlim([-preTime,postTime])
        ax.set_ylim([-40,40])
        ax.set_xlabel('time from saccade onset (s)')
        ax.set_ylabel('pupil X position')
        ax.set_title(lbl)
    plt.tight_layout()
    plt.savefig(os.path.join(outputPath,'saccades','saccades',str(sessionID)+'_saccades.png'))
    plt.close(fig)
    
    
    channels = session.get_channels()
    units = session.get_units()
    spikeTimes = session.spike_times
    good_unit_filter = ((units['snr']>1)&(units['isi_violations']<1) & (units['firing_rate']>0.1))
    units = units.loc[good_unit_filter]
    unitchannels = units.merge(channels, left_on='peak_channel_id', right_index=True)
    
    for region in ('APN','VISp'):
        negSaccadeSdfs = []
        posSaccadeSdfs = []
        preTime = 0.5
        postTime = 1
        for u,unit in unitchannels.iterrows():
            if unit['manual_structure_acronym'] == region:
                for saccades,sdfs in zip((negSaccades,posSaccades),(negSaccadeSdfs,posSaccadeSdfs)):
                    sdf,tsdf = getSdf(spikeTimes[u],t[saccades]-preTime,preTime+postTime)
                    sdfs.append(sdf)
        
        if len(negSaccadeSdfs) > 0:
            fig = plt.figure(figsize=(6,6))
            for ind,(sdfs,clr,lbl) in enumerate(zip((negSaccadeSdfs,posSaccadeSdfs),'rb',('negative saccades','positive saccades'))):
                ax = fig.add_subplot(2,1,ind+1)
                sdfs = np.array(sdfs)
                sdfs -= sdfs[:,tsdf<0.2].mean(axis=1)[:,None]
                for s in sdfs:
                    ax.plot(tsdf-preTime,s,color=clr,alpha=0.2)
                for side in ('right','top'):
                    ax.spines[side].set_visible(False)
                ax.tick_params(direction='out',top=False,right=False)
                if ind==1:
                    ax.set_xlabel('time from saccade (s)')
                ax.set_ylabel('spikes/s')
                ax.set_title(region + ' (' + lbl + ')')
            plt.tight_layout()
            plt.savefig(os.path.join(outputPath,'saccades','sdfs',str(sessionID)+'_sdfs_'+region+'.png'))
            plt.close(fig)



prevLickErrorSessions = []

prevEyeErrorSessions = []


lickErrorSessions = []
eyeErrorSessions = []
for nwbInd,nwbPath in enumerate(nwb_paths):
    print(str(nwbInd+1) + ' / ' + str(len(nwb_paths)))
    
    prevErrorSessions = prevLickErrorSessions + prevEyeErrorSessions
    if len(prevErrorSessions) > 0:
        sid = int(nwbPath[-14:-4])
        if sid not in prevErrorSessions:
            continue
    
    with NWBHDF5IO(nwbPath, 'r', load_namespaces=True) as nwb_io:
        session = BehaviorEcephysSession.from_nwb(nwbfile=nwb_io.read())
    sessionID = session.metadata['ecephys_session_id']
    
    if len(prevErrorSessions)==0 or sessionID in prevLickErrorSessions:    
        try:
            makeLickPlots(session)
        except:
            lickErrorSessions.append(sessionID)
    
    if len(prevErrorSessions)==0 or sessionID in prevEyeErrorSessions:
        try:
            makeSaccadePlots(session)
        except:
            eyeErrorSessions.append(sessionID)

    plt.close('all')
    gc.collect()

print('lick errors')
print(lickErrorSessions)
print('eye errors')
print(eyeErrorSessions)
