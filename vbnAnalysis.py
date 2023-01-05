# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 16:22:30 2022

@author: svc_ccg
"""

import copy
import os
import numpy as np
import scipy.stats
import pandas as pd
import h5py
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42



## make h5df with binned spike counts
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
nBins = int(windowDur/binSize)

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
    
    h5Group = h5File.create_group(str(sessionId))
    h5Group.create_dataset('unitIds',data=goodUnits.index,compression='gzip',compression_opts=4)
    spikes = h5Group.create_dataset('spikes',shape=(len(goodUnits),len(flashTimes),nBins),dtype=bool,chunks=(1,len(flashTimes),nBins),compression='gzip',compression_opts=4)
    
    i = 0
    for unitId,unitData in goodUnits.iterrows(): 
        spikes[i] = getSpikeBins(spikeTimes[unitId],flashTimes,windowDur,binSize)
        i += 1

h5File.close()
    


##
baseDir = r"C:\Users\svc_ccg\Desktop\Analysis\vbn"

stimTable = pd.read_csv(os.path.join(baseDir,'master_stim_table.csv'))

unitTable = pd.read_csv(os.path.join(baseDir,'units_with_cortical_layers.csv'))

unitData = h5py.File(os.path.join(baseDir,'vbnAllUnitSpikeTensor.hdf5'),mode='r')


sessionIds = stimTable['session_id'][stimTable['experience_level']=='Familiar'].unique()

regions = ('LGd','VISp','VISl','VISrl','VISal','VISpm','VISam','LP','MRN')
layers = (('1','2/3'),'4','5',('6a','6b'))

binSize = 0.001
baseWin = slice(680,750)
respWin = slice(30,100)


def findResponsiveUnits(basePsth,respPsth,baseWin,respWin):
    hasSpikes = ((respPsth[:,:,respWin].mean(axis=(1,2)) - basePsth[:,:,baseWin].mean(axis=(1,2))) / 0.001) > 0.1
    
    base = basePsth[:,:,baseWin].mean(axis=1)
    resp = respPsth[:,:,respWin].mean(axis=1)
    peak = np.max(resp-base.mean(axis=1)[:,None],axis=1)
    hasPeakResp = peak > 5 * base.std()
    
    base = basePsth[:,:,baseWin].mean(axis=2)
    resp = respPsth[:,:,respWin].mean(axis=2)
    pval = np.array([1 if np.sum(r-b)==0 else scipy.stats.wilcoxon(b,r)[1] for b,r in zip(base,resp)])
    
    return hasSpikes & hasPeakResp & (pval<0.05)

    
#
unitCount = np.zeros((len(sessionIds),len(regions)),dtype=int)
for i,sid in enumerate(sessionIds):
    units = unitTable.set_index('unit_id').loc[unitData[str(sid)]['unitIds'][:]]
    for j,reg in enumerate(regions):
        unitCount[i,j] = np.sum(units['structure_acronym']==reg)


#
changeSpikes = {region: {layer: [] for layer in layers} for region in regions}
preChangeSpikes = copy.deepcopy(changeSpikes)  
changeResp = copy.deepcopy(changeSpikes)
preChangeResp = copy.deepcopy(changeSpikes)  
adaptSpikes = copy.deepcopy(changeSpikes)
adaptResp = copy.deepcopy(changeSpikes)
flashBase = copy.deepcopy(changeSpikes)
flashResp = copy.deepcopy(changeSpikes)
changeFlashBase = copy.deepcopy(changeSpikes)
changeFlashResp = copy.deepcopy(changeSpikes)
for si,sid in enumerate(sessionIds):
    units = unitTable.set_index('unit_id').loc[unitData[str(sid)]['unitIds'][:]]
    spikes = unitData[str(sid)]['spikes']
    
    stim = stimTable[(stimTable['session_id']==sid) & stimTable['active']].reset_index()
    autoRewarded = np.array(stim['auto_rewarded']).astype(bool)
    changeFlash = np.where(stim['is_change'] & ~autoRewarded)[0]
    notOmitted = [flash for flash in changeFlash if not any(stim[flash-1:flash+10]['omitted']) and flash+10<spikes.shape[1]]
    
    for region in regions:
        inRegion = np.array(units['structure_acronym']==region)
        if not any(inRegion):
            continue
        for layer in layers:
            print('session '+str(si+1)+', '+region+', '+str(layer))
            if 'VIS' in region:
                inLayer = inRegion & np.in1d(units['cortical_layer'],layer)
            elif '1' not in layer:
                break
            else:
                inLayer = inRegion
            if not any(inLayer):
                continue
            s = np.zeros((inLayer.sum(),spikes.shape[1],spikes.shape[2]),dtype=bool)
            for i,u in enumerate(np.where(inLayer)[0]):
                s[i]=spikes[u,:,:]
                
            changeSp = s[:,changeFlash,:]
            preChangeSp = s[:,changeFlash-1,:]
            hasResp = findResponsiveUnits(preChangeSp,changeSp,baseWin,respWin)
            if not any(hasResp):
                continue
            base = s[hasResp,:,baseWin].sum(axis=2)
            resp = np.full((hasResp.sum(),len(stim)),np.nan)
            resp[:,1:] = s[hasResp,1:,respWin].sum(axis=2) - base[:,:-1]
            
            changeSpikes[region][layer].append(changeSp[hasResp].mean(axis=1))
            preChangeSpikes[region][layer].append(preChangeSp[hasResp].mean(axis=1))
            changeResp[region][layer].append(resp[:,changeFlash])
            preChangeResp[region][layer].append(resp[:,changeFlash-1])
            
            adaptSp = np.zeros((hasResp.sum(),len(notOmitted),11*750),dtype=bool)
            adaptR = np.zeros((hasResp.sum(),len(notOmitted),11))
            for i,flash in enumerate(notOmitted):
                adaptSp[:,i,:] = s[hasResp,flash-1:flash+10,:].reshape((hasResp.sum(),-1))
                adaptR[:,i,:] = resp[:,flash-1:flash+10]
            adaptSpikes[region][layer].append(adaptSp.mean(axis=1))
            adaptResp[region][layer].append(adaptR.mean(axis=1))
            
            flashCount = np.zeros(12)
            fb = np.zeros((hasResp.sum(),12))
            fr = fb.copy()
            changeCount = flashCount.copy()
            cb = fb.copy()
            cr = fb.copy()
            for i,row in stim.iterrows():
                lastLick = row['flashes_since_last_lick']
                if not np.isnan(lastLick) and lastLick<13:
                    ind = int(lastLick)-1
                    if not row['previous_omitted'] and lastLick < row['flashes_since_change']:
                        flashCount[ind] += 1
                        fb[:,ind] += base[:,i-1]
                        fr[:,ind] += resp[:,i]
                    if row['is_change'] and not row['auto_rewarded']:
                        changeCount[ind] += 1
                        cb[:,ind] += base[:,i-1]
                        cr[:,ind] += resp[:,i]
            fb /= flashCount
            fr /= flashCount
            cb /= changeCount
            cr /= changeCount
            flashBase[region][layer].append(fb)
            flashResp[region][layer].append(fr)
            changeFlashBase[region][layer].append(cb)
            changeFlashResp[region][layer].append(cr)
            
                             

t = np.arange(11*750)/1000 - 0.75
flashTimes = np.arange(-0.75,7.5,0.75)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for region,clr in zip(regions,plt.cm.magma(np.linspace(0,0.8,len(regions)))):
    if 'VIS' in region:
        d = np.concatenate([np.concatenate(adaptSpikes[region][layer]) for layer in layers if len(adaptSpikes[region][layer])>0])
    else:
        d = np.concatenate(adaptSpikes[region][layers[0]])
    d -= d[:,baseWin].mean(axis=1)[:,None]
    d /= binSize
    ax.plot(t,d.mean(axis=0),color=clr,alpha=0.5,label=region+', n='+str(d.shape[0]))            
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlim([-0.25,7.5])
ax.set_xlabel('Time from change (s)')
ax.set_ylabel('Spikes/s')
ax.legend(loc='upper center')
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for region,clr in zip(regions,plt.cm.magma(np.linspace(0,0.8,len(regions)))):
    if 'VIS' in region:
        d = np.concatenate([np.concatenate(adaptResp[region][layer]) for layer in layers if len(adaptSpikes[region][layer])>0])
    else:
        d = np.concatenate(adaptResp[region][layers[0]])
    d /= d[:,1][:,None]
    mean = d.mean(axis=0)
    sem = d.std(axis=0)/(d.shape[0]**0.5)
    ax.plot(flashTimes,mean,color=clr,alpha=0.5,label=region)
    for x,m,s in zip(flashTimes,mean,sem):
        ax.plot([x,x],[m-s,m+s],color=clr,alpha=0.5)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlim([-1,7.5])
ax.set_ylim([0,1.01])
ax.set_xlabel('Time from change (s)')
ax.set_ylabel('Norm. response')
ax.legend(loc='lower right')
plt.tight_layout()

fig = plt.figure(figsize=(6,8))
xticks = np.arange(len(regions))
for i,layer in enumerate(layers):
    ax = fig.add_subplot(4,1,i+1)
    mean = []
    sem = []
    for region in regions:
        if 'VIS' in region:
            d = np.concatenate(adaptResp[region][layer])
        else:
            d = np.concatenate(adaptResp[region][layers[0]])
        d /= d[:,1][:,None]
        mean.append(d[:,-1].mean(axis=0))
        sem.append(d[:,-1].std(axis=0)/(d.shape[0]**0.5))
    ax.plot(xticks,mean,color='k')
    for x,m,s in zip(xticks,mean,sem):
        ax.plot([x,x],[m-s,m+s],color='k')
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xticks(xticks)
    ax.set_xticklabels(regions)
    ax.set_ylim([0,1.01])
    ax.set_ylabel('Adaptation ratio')
    ax.set_title('cortical layer '+str(layer))
plt.tight_layout()

fig = plt.figure(figsize=(6,11))
for i,region in enumerate(r for r in regions if 'VIS' in r):
    ax = fig.add_subplot(6,1,i+1)
    for layer,clr in zip(layers,plt.cm.magma(np.linspace(0,0.8,len(layers)))):
        if len(adaptSpikes[region][layer])>0:
            d = np.concatenate(adaptSpikes[region][layer])
            d -= d[:,baseWin].mean(axis=1)[:,None]
            d /= binSize
            ax.plot(t,d.mean(axis=0),color=clr,alpha=0.5,label=str(layer)+', n='+str(d.shape[0]))            
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xlim([-0.25,7.5])
    if i==5:
        ax.set_xlabel('Time from change (s)')
    ax.set_ylabel('Spikes/s')
    ax.legend(loc='upper center')
    ax.set_title(region)
plt.tight_layout()

fig = plt.figure(figsize=(6,11))
for i,region in enumerate(r for r in regions if 'VIS' in r):
    ax = fig.add_subplot(6,1,i+1)
    for layer,clr in zip(layers,plt.cm.magma(np.linspace(0,0.8,len(layers)))):
        if len(adaptSpikes[region][layer])>0:
            d = np.concatenate(adaptResp[region][layer])
            d /= d[:,1][:,None]
            mean = d.mean(axis=0)
            sem = d.std(axis=0)/(d.shape[0]**0.5)
            ax.plot(flashTimes,mean,color=clr,alpha=0.5,label=layer)
            for x,m,s in zip(flashTimes,mean,sem):
                ax.plot([x,x],[m-s,m+s],color=clr,alpha=0.5)            
    for side in ('right','top'):
        ax.spines[side].set_visible(False)
    ax.tick_params(direction='out',top=False,right=False)
    ax.set_xlim([-1,7.5])
    ax.set_ylim([0,1.01])
    if i==5:
        ax.set_xlabel('Time from change (s)')
    if i==0:
        ax.set_ylabel('Norm. response')
        ax.legend(loc='upper center')
    ax.set_title(region)
plt.tight_layout()



flashSinceLickTimes = np.arange(0.75,0.75*13,0.75)
for r,ylbl in zip((flashResp,flashBase,changeFlashResp,changeFlashBase),('flash resp','pre-flash baseline','change resp','pre-change baseline')):
    fig = plt.figure(figsize=(6,8))
    for i,layer in enumerate(layers):
        ax = fig.add_subplot(4,1,i+1)
        ymax = 0
        for region,clr in zip(regions,plt.cm.magma(np.linspace(0,0.8,len(regions)))):
            if 'VIS' in region:
                d = np.concatenate(r[region][layer])
            else:
                d = np.concatenate(r[region][layers[0]])
            d /= (respWin.stop-respWin.start)/1000
            mean = np.nanmean(d,axis=0)
            sem = np.nanstd(d,axis=0)/(d.shape[0]**0.5)
            lbl = region if i==0 else None
            ax.plot(flashSinceLickTimes,mean,color=clr,alpha=0.5,label=lbl)
            for x,m,s in zip(flashSinceLickTimes,mean,sem):
                ax.plot([x,x],[m-s,m+s],color=clr,alpha=0.5)
            ymax = max(ymax,np.nanmax(mean+sem))
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        ax.set_xlim([0,9.75])
        ax.set_ylim([0,ymax])
        ax.set_xlabel('time since lick (s)')
        ax.set_ylabel(ylbl+'\n(spikes/s)')
        if i==0:
            loc = 'upper left' if 'change' in ylbl else 'upper right'
            ax.legend(loc=loc)
        ax.set_title('cortical layer '+str(layer))
    plt.tight_layout()











