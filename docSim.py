# -*- coding: utf-8 -*-
"""
Created on Thu Jun 30 15:27:03 2022
@author: svc_ccg
"""

import random
import numpy as np
import matplotlib.pyplot as plt
from DynamicRoutingAnalysisUtils import calcDprime,adjustResponseRate

    
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
            lick = False
            if random.random() < self.timingProb:
                if trialFlash <= len(self.lickProbTiming) and random.random() < self.lickProbTiming[trialFlash-1]:
                    lick = True
            elif isChange and random.random() < self.lickProbChange:
                lick = True
            elif random.random() < self.lickProb:
                lick = True
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
        hit,miss,fa,cr = [[outcome==label for outcome in self.trialOutcome] for label in ('hit','miss','false alarm','correct reject')]
        self.rewardRate = sum(hit) / hours
        goTrials = sum(hit) + sum(miss)
        nogoTrials = sum(fa) + sum(cr)
        if goTrials==0 or nogoTrials==0:
            self.dprime = np.nan
        else:
            hitRate = sum(hit) / goTrials
            falseAlarmRate = sum(fa) / nogoTrials
            self.dprime = calcDprime(hitRate,falseAlarmRate,goTrials,nogoTrials)
        
            
            
def pickChangeFlash(p=0.3,nmin=5,nmax=12):
    n = np.random.geometric(p)
    if nmin <= n <= nmax:
        return n
    else:
        return pickChangeFlash()
    

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
lickProbTiming = np.zeros(13)
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




            
            
            
def gamma(x,alpha,tau,eta):
    return ((alpha*(x-tau))**eta * np.exp(-alpha*(x-tau))) / (eta**eta * np.exp(-eta))


x = np.arange(1,13)
alpha = [1]
tau = [2,4,8]
eta = [1,2,4]


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for a in alpha:
    for t in tau:
        for e in eta:
            y = gamma(x,a,t,e)
            ax.plot(x,y,label=(a,t,e))
ax.set_ylim([0,1])
ax.legend(fontsize=6)


import scipy.stats

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for a in [6]:
    for s in [0]:
        y = scipy.stats.gamma.pdf(x,a,scale=s)
        ax.plot(x,y/y.max(),label=(a,s))
ax.legend(fontsize=6)


        

doc = DocSim()

doc.runSession(sessionHours=100)


isChange = ~np.isnan(doc.changeFlash)
isCatch = ~np.isnan(doc.catchFlash)
isAbort = doc.trialOutcome == 'abort'

flashesToChange = np.array(doc.changeFlash)[isChange] - np.array(doc.trialStartFlash)[isChange] + 1
flashesToCatch = np.array(doc.catchFlash)[isCatch] - np.array(doc.trialStartFlash)[isCatch] + 1
flashesToAbort = np.array(doc.)

flashNum = np.arange(1,flashesToChange.max()+1)

changeProb = np.array([(np.sum(flashesToChange==n) - np.sum(flashesToCatch==n)) / (flashesToChange.size + flashesToCatch.size) for n in flashNum])

changeProb = np.array([np.sum(flashesToChange==n)/(flashesToChange.size/(1-doc.catchProb)) for n in flashNum])






n = np.array([getChangeFlash() for _ in range(100000)])

flashNum = np.arange(1,n.max()+1)

changeProb = np.array([np.sum(n==i)/(n.size/(1-catchProb)) for i in flashNum])

conditionalChangeProb = np.array([changeProb[i]/(changeProb[i:].sum()+catchProb) for i in range(n.max())])

abortProb = np.zeros(flashNum[-1])
for i in range(1,flashNum[-2]):
    abortProb[i+1] = (1-abortProb[i-1])*conditionalChangeProb[i]

changeProbWithAborts = changeProb * (1-abortProb)

predictedLickProb = changeProbWithAborts * conditionalChangeProb

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot([0,flashNum[-1]],[catchProb]*2,'--',color='0.5',label='catch prob.')
ax.plot(flashNum,changeProb,'ko-',label='change prob. (including catch trials)')
ax.plot(flashNum,conditionalChangeProb,'bo-',label='change prob. given no change previous flash')
ax.plot(flashNum,changeProbWithAborts,'ro-',label='change prob with predicted aborts')
ax.plot(flashNum,predictedLickProb,'g-',label='predicted lick prob. (blue * red)')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlabel('flash number')
ax.set_ylabel('probability')
ax.legend(bbox_to_anchor=(0.5,1),fontsize=8)
plt.tight_layout()



