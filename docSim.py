# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 15:27:03 2022
@author: svc_ccg
"""

import random
import numpy as np
import scipy.stats
import matplotlib.pyplot as plt

    
class DocSim():
    
    def __init__(self,lickProb=0,lickProbChange=0,lickProbTiming=[0],timingProb=0):
        self.lickProb = lickProb
        self.lickProbChange = lickProbChange
        self.lickProbTiming = lickProbTiming
        self.timingProb = timingProb
        self.flashInterval = 0.75 # seconds
        self.catchProb = 0.125
        self.timeoutFlashes = 0
        self.graceFlashes = 5
        self.maxAborts = 5
    
    def runTrial(self):
        self.trialStartFlash.append(self.flash+1)
        if self.aborts <= self.maxAborts:
            self.trialChangeFlash = pickChangeFlash()
            
        lick = False
        outcome = False
        for trialFlash in range(1,self.trialChangeFlash+1):
            self.flash += 1
            if trialFlash == self.trialChangeFlash:
                if random.random() < self.catchProb:
                    isCatch,isChange = True,False
                else:
                    isCatch,isChange = False,True
            else:
                isCatch = isChange = False
            
            if random.random() < self.timingProb and trialFlash <= len(self.lickProbTiming) and random.random() < self.lickProbTiming[trialFlash-1]:
                lick = True
            elif isChange and random.random() < self.lickProbChange:
                lick = True
            elif random.random() < self.lickProb:
                lick = True
            else:
                lick = False
            
            if isChange:
                outcome = 'hit' if lick else 'miss'
            elif isCatch:
                outcome = 'false alarm' if lick else 'correct reject'
            elif lick:
                outcome = 'abort'
            
            if outcome:
                self.trialOutcomeFlash.append(self.flash)
                self.trialOutcome.append(outcome)
                if outcome == 'abort':
                    if self.aborts < self.maxAborts:
                        self.aborts += 1
                    else:
                        self.aborts = 0
                    self.flash += self.timeoutFlashes
                else:
                    self.aborts = 0
                    self.flash += self.graceFlashes
                break
                
    def runSession(self,sessionHours):
        self.flash = 0
        self.aborts = 0
        self.trialStartFlash = []
        self.trialOutcomeFlash = []
        self.trialOutcome = []
        hours = 0
        while hours < sessionHours:
            self.runTrial()
            hours = self.flash * self.flashInterval / 3600
        hits,misses,falseAlarms,correctRejects = [sum(outcome==label for outcome in self.trialOutcome) for label in ('hit','miss','false alarm','correct reject')]
        self.rewardRate = hits / hours
        self.dprime = calcDprime(hits,misses,falseAlarms,correctRejects)
        
            
            
def pickChangeFlash(p=0.3,nmin=5,nmax=12):
    n = np.random.geometric(p)
    if nmin <= n <= nmax:
        return n
    else:
        return pickChangeFlash()
    

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
    

# calculate change prob
catchProb = 0.125    
n = np.array([pickChangeFlash() for _ in range(100000)])
flashNum = np.arange(1,n.max()+1)
changeProb = np.array([np.sum(n==i)/n.size for i in flashNum]) * (1-catchProb)
conditionalChangeProb = np.array([changeProb[i]/(changeProb[i:].sum()/(1-catchProb)) for i in range(n.max())])

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(flashNum,changeProb,'ko-',label='change prob.')
ax.plot(flashNum,conditionalChangeProb,'bo-',label='change prob. given no change previous flash')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_ylim([0,1])
ax.set_xlabel('flash number')
ax.set_ylabel('probability')
ax.legend(loc='upper left',fontsize=8)
plt.tight_layout()


# random licking
doc = DocSim(lickProb=0,lickProbChange=1)
doc.runSession(sessionHours=10)

p = np.arange(0,1.05,0.1)
rewardRate = np.zeros((p.size,)*2)
dprime = rewardRate.copy()
for i,lickProb in enumerate(p):
    for j,lickProbChange in enumerate(p):
        print(i,j)
        doc = DocSim(lickProb,lickProbChange)
        doc.runSession(sessionHours=10)
        rewardRate[i,j] = doc.rewardRate
        dprime[i,j] = doc.dprime
        
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
im = ax.imshow(rewardRate,clim=[0,rewardRate.max()],cmap='gray',origin='lower')
ax.set_xticks([0,rewardRate.shape[1]-1])
ax.set_xticklabels([0,1])
ax.set_xlabel('change lick prob')
ax.set_yticks([0,rewardRate.shape[0]-1])
ax.set_yticklabels([0,1])
ax.set_ylabel('random lick prob')
cb = plt.colorbar(im,ax=ax,fraction=0.04,pad=0.04)
ax.set_title('rewards/hour')
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
cmax = max(np.absolute(np.nanmin(dprime)),np.nanmax(dprime))
d = np.ma.array(dprime,mask=np.isnan(dprime))
cmap = plt.cm.bwr
cmap.set_bad('k')
im = ax.imshow(d,clim=[-cmax,cmax],cmap=cmap,origin='lower')
ax.set_xticks([0,dprime.shape[1]-1])
ax.set_xticklabels([0,1])
ax.set_xlabel('change lick prob')
ax.set_yticks([0,dprime.shape[0]-1])
ax.set_yticklabels([0,1])
ax.set_ylabel('random lick prob')
cb = plt.colorbar(im,ax=ax,fraction=0.04,pad=0.04)
ax.set_title('d prime')
plt.tight_layout()


# lick on 5th flash
lickProbTiming = np.zeros(12)
lickProbTiming[4] = 1     
p = np.arange(0,1.05,0.1)
rewardRate = np.zeros((p.size,)*2)
dprime = rewardRate.copy()
for i,timingProb in enumerate(p):
    for j,lickProbChange in enumerate(p):
        print(i,j)
        doc = DocSim(0,lickProbChange,lickProbTiming,timingProb)
        doc.runSession(sessionHours=10)
        rewardRate[i,j] = doc.rewardRate
        dprime[i,j] = doc.dprime

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
im = ax.imshow(rewardRate,clim=[0,rewardRate.max()],cmap='gray',origin='lower')
ax.set_xticks([0,rewardRate.shape[1]-1])
ax.set_xticklabels([0,1])
ax.set_xlabel('change lick prob')
ax.set_yticks([0,rewardRate.shape[0]-1])
ax.set_yticklabels([0,1])
ax.set_ylabel('timing prob')
cb = plt.colorbar(im,ax=ax,fraction=0.04,pad=0.04)
ax.set_title('rewards/hour')
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
cmax = max(np.absolute(np.nanmin(dprime)),np.nanmax(dprime))
d = np.ma.array(dprime,mask=np.isnan(dprime))
cmap = plt.cm.bwr
cmap.set_bad('k')
im = ax.imshow(d,clim=[-cmax,cmax],cmap=cmap,origin='lower')
ax.set_xticks([0,dprime.shape[1]-1])
ax.set_xticklabels([0,1])
ax.set_xlabel('change lick prob')
ax.set_yticks([0,dprime.shape[0]-1])
ax.set_yticklabels([0,1])
ax.set_ylabel('timing prob')
cb = plt.colorbar(im,ax=ax,fraction=0.04,pad=0.04)
ax.set_title('d prime')
plt.tight_layout()


# lick accoring to conditional change prob   
p = np.arange(0,1.05,0.1)
rewardRate = np.zeros((p.size,)*2)
dprime = rewardRate.copy()
for i,timingProb in enumerate(p):
    for j,lickProbChange in enumerate(p):
        print(i,j)
        doc = DocSim(0,lickProbChange,changeProb/changeProb.max(),timingProb)
        doc.runSession(sessionHours=10)
        rewardRate[i,j] = doc.rewardRate
        dprime[i,j] = doc.dprime

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
im = ax.imshow(rewardRate,clim=[0,rewardRate.max()],cmap='gray',origin='lower')
ax.set_xticks([0,rewardRate.shape[1]-1])
ax.set_xticklabels([0,1])
ax.set_xlabel('change lick prob')
ax.set_yticks([0,rewardRate.shape[0]-1])
ax.set_yticklabels([0,1])
ax.set_ylabel('timing prob')
cb = plt.colorbar(im,ax=ax,fraction=0.04,pad=0.04)
ax.set_title('rewards/hour')
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
cmax = max(np.absolute(np.nanmin(dprime)),np.nanmax(dprime))
d = np.ma.array(dprime,mask=np.isnan(dprime))
cmap = plt.cm.bwr
cmap.set_bad('k')
im = ax.imshow(d,clim=[-cmax,cmax],cmap=cmap,origin='lower')
ax.set_xticks([0,dprime.shape[1]-1])
ax.set_xticklabels([0,1])
ax.set_xlabel('change lick prob')
ax.set_yticks([0,dprime.shape[0]-1])
ax.set_yticklabels([0,1])
ax.set_ylabel('timing prob')
cb = plt.colorbar(im,ax=ax,fraction=0.04,pad=0.04)
ax.set_title('d prime')
plt.tight_layout()


# lick accoring to conditional change prob   
p = np.arange(0,1.05,0.1)
rewardRate = np.zeros((p.size,)*2)
dprime = rewardRate.copy()
for i,timingProb in enumerate(p):
    for j,lickProbChange in enumerate(p):
        print(i,j)
        doc = DocSim(0,lickProbChange,conditionalChangeProb,timingProb)
        doc.runSession(sessionHours=10)
        rewardRate[i,j] = doc.rewardRate
        dprime[i,j] = doc.dprime

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
im = ax.imshow(rewardRate,clim=[0,rewardRate.max()],cmap='gray',origin='lower')
ax.set_xticks([0,rewardRate.shape[1]-1])
ax.set_xticklabels([0,1])
ax.set_xlabel('change lick prob')
ax.set_yticks([0,rewardRate.shape[0]-1])
ax.set_yticklabels([0,1])
ax.set_ylabel('timing prob')
cb = plt.colorbar(im,ax=ax,fraction=0.04,pad=0.04)
ax.set_title('rewards/hour')
plt.tight_layout()

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
cmax = max(np.absolute(np.nanmin(dprime)),np.nanmax(dprime))
d = np.ma.array(dprime,mask=np.isnan(dprime))
cmap = plt.cm.bwr
cmap.set_bad('k')
im = ax.imshow(d,clim=[-cmax,cmax],cmap=cmap,origin='lower')
ax.set_xticks([0,dprime.shape[1]-1])
ax.set_xticklabels([0,1])
ax.set_xlabel('change lick prob')
ax.set_yticks([0,dprime.shape[0]-1])
ax.set_yticklabels([0,1])
ax.set_ylabel('timing prob')
cb = plt.colorbar(im,ax=ax,fraction=0.04,pad=0.04)
ax.set_title('d prime')
plt.tight_layout()




