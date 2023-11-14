# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 14:39:37 2023

@author: svc_ccg
"""

import os
import math
import numpy as np
import scipy.stats
import scipy.cluster
import sklearn
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42


def dictToHdf5(group,d):
    for key,val in d.items():
        if isinstance(val,dict): 
            dictToHdf5(group.create_group(key),val)
        else:
            group.create_dataset(key,data=val)


def findNearest(array,values):
    ind = np.searchsorted(array,values,side='left')
    for i,j in enumerate(ind):
        if j > 0 and (j == len(array) or math.fabs(values[i] - array[j-1]) < math.fabs(values[i] - array[j])):
            ind[i] = j-1
    return ind


def getBehavData(stim):
    # stim = stimulus table or index of
    flashTimes = np.array(stim['start_time'])
    changeTimes = flashTimes[stim['is_change']]
    hit = np.array(stim['hit'])
    engaged = np.array([np.sum(hit[stim['is_change']][(changeTimes>t-60) & (changeTimes<t+60)]) > 1 for t in flashTimes])
    autoRewarded = np.array(stim['auto_rewarded']).astype(bool)
    changeFlashes = np.array(stim['is_change'] & ~autoRewarded & engaged)
    catch = stim['catch'].copy()
    catch[catch.isnull()] = False
    catch = np.array(catch).astype(bool) & engaged
    catchFlashes = np.zeros(catch.size,dtype=bool)
    catchFlashes[np.searchsorted(flashTimes,np.unique(stim['change_time_no_display_delay'][catch]))] = True
    omittedFlashes = np.array(stim['omitted']) & engaged
    prevOmittedFlashes = np.array(stim['previous_omitted']) & engaged
    nonChangeFlashes = np.array(engaged &
                                (~stim['is_change']) & 
                                (~stim['omitted']) & 
                                (~stim['previous_omitted']) & 
                                (stim['flashes_since_change']>5) &
                                (stim['flashes_since_last_lick']>1))
    novelFlashes = stim['novel_image'].copy()
    novelFlashes[novelFlashes.isnull()] = False
    novelFlashes = np.array(novelFlashes).astype(bool) & engaged
    lick = np.array(stim['lick_for_flash'])
    lickTimes = np.array(stim['lick_time'])
    lickLatency = lickTimes - flashTimes
    earlyLick = lickLatency < 0.15
    lateLick = lickLatency > 0.75
    lick[earlyLick | lateLick] = False
    lickTimes[earlyLick | lateLick] = np.nan
    return flashTimes,changeFlashes,catchFlashes,nonChangeFlashes,omittedFlashes,prevOmittedFlashes,novelFlashes,lick,lickTimes


def getUnitsInRegion(units,region,rs=False,fs=False):
    if region in ('SC/MRN cluster 1','SC/MRN cluster 2'):
        clust = 1 if '1' in region else 2
        dirPath = '/allen/programs/mindscope/workgroups/np-behavior/VBN_video_analysis'
        clustId = np.load(os.path.join(dirPath,'sc_mrn_clusterId.npy'))
        clustUnitId = np.load(os.path.join(dirPath,'sc_mrn_clusterUnitId.npy'))
        u = clustUnitId[np.in1d(clustUnitId,units.index) & (clustId==clust)]
        inRegion = np.in1d(units.index,u)
    else:
        if 'weighted' in region:
            region = region[:region.find(' weighted')]
        if region=='VISall':
            reg = ('VISp','VISl','VISrl','VISal','VISpm','VISam')
        elif region=='SC':
            reg = ('SCig','SCiw')
        elif region=='Hipp':
            reg = ('HPF','DG','CA1','CA3')
        elif region=='Sub':
            reg = ('SUB','ProS','PRE','POST')
        else:
            reg = region
        inRegion = np.in1d(units['structure_acronym'],reg)
        if 'VIS' in region:
            rsUnits = np.array(units['waveform_duration'] > 0.4)
            if rs and not fs:
                inRegion = inRegion & rsUnits
            elif fs and not rs:
                inRegion = inRegion & ~rsUnits
    return inRegion
    

def findResponsiveUnits(basePsth,respPsth,baseWin,respWin):
    hasSpikes = ((respPsth[:,:,respWin].mean(axis=(1,2)) - basePsth[:,:,baseWin].mean(axis=(1,2))) / 0.001) > 0.1
    
    base = basePsth[:,:,baseWin].mean(axis=1)
    resp = respPsth[:,:,respWin].mean(axis=1)
    peak = np.max(resp-base.mean(axis=1)[:,None],axis=1)
    hasPeakResp = peak > 5 * base.std(axis=1)
    
    base = basePsth[:,:,baseWin].mean(axis=2)
    resp = respPsth[:,:,respWin].mean(axis=2)
    pval = np.array([1 if np.sum(r-b)==0 else scipy.stats.wilcoxon(b,r)[1] for b,r in zip(base,resp)])
    
    return hasSpikes & hasPeakResp & (pval<0.05)


def trainDecoder(model,X,y,nSplits):
    # cross validation using stratified shuffle split
    # each split preserves the percentage of samples of each class
    # all samples used in one test set
    
    classVals = np.unique(y)
    nClasses = len(classVals)
    nSamples = len(y)
    samplesPerClass = [np.sum(y==val) for val in classVals]
    samplesPerSplit = [round(n/nSplits) for n in samplesPerClass]
    shuffleInd = np.random.permutation(nSamples)
    
    cv = {'estimator': [sklearn.base.clone(model) for _ in range(nSplits)]}
    cv['train_score'] = []
    cv['test_score'] = []
    cv['predict'] = np.full(nSamples,np.nan)
    cv['predict_proba'] = np.full((nSamples,nClasses),np.nan)
    cv['decision_function'] = np.full((nSamples,nClasses),np.nan) if nClasses>2 else np.full(nSamples,np.nan)
    cv['feature_importance'] = []
    cv['coef'] = []
    modelMethods = dir(model)
    
    for k,estimator in enumerate(cv['estimator']):
        testInd = []
        for val,n in zip(classVals,samplesPerSplit):
            start = k*n
            ind = shuffleInd[y[shuffleInd]==val]
            testInd.extend(ind[start:start+n] if k+1<nSplits else ind[start:])
        trainInd = np.setdiff1d(shuffleInd,testInd)
        estimator.fit(X[trainInd],y[trainInd])
        cv['train_score'].append(estimator.score(X[trainInd],y[trainInd]))
        cv['test_score'].append(estimator.score(X[testInd],y[testInd]))
        cv['predict'][testInd] = estimator.predict(X[testInd])
        for method in ('predict_proba','decision_function'):
            if method in modelMethods:
                cv[method][testInd] = getattr(estimator,method)(X[testInd])
        for attr in ('feature_importance_','coef_'):
            if attr in estimator.__dict__:
                cv[attr[:-1]].append(getattr(estimator,attr))
    return cv


def pca(data,plot=False):
    # data is n samples x m parameters
    eigVal,eigVec = np.linalg.eigh(np.cov(data,rowvar=False))
    order = np.argsort(eigVal)[::-1]
    eigVal = eigVal[order]
    eigVec = eigVec[:,order]
    pcaData = data.dot(eigVec)
    if plot:
        fig = plt.figure(facecolor='w')
        ax = fig.add_subplot(1,1,1)
        ax.plot(np.arange(1,eigVal.size+1),eigVal.cumsum()/eigVal.sum(),'k')
        ax.set_xlim((0.5,eigVal.size+0.5))
        ax.set_ylim((0,1.02))
        ax.set_xlabel('PC')
        ax.set_ylabel('Cumulative Fraction of Variance')
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        
        fig = plt.figure(facecolor='w')
        ax = fig.add_subplot(1,1,1)
        im = ax.imshow(eigVec,clim=(-1,1),cmap='bwr',interpolation='none',origin='lower')
        ax.set_xlabel('PC')
        ax.set_ylabel('Parameter')
        ax.set_title('PC Weightings')
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        cb = plt.colorbar(im,ax=ax,fraction=0.05,pad=0.04,shrink=0.5)
        cb.ax.tick_params(length=0)
        cb.set_ticks([-1,0,1])
    return pcaData,eigVal,eigVec


def cluster(data,nClusters=None,method='ward',metric='euclidean',plot=False,colors=None,labels=None,xmax=None,nreps=1000,title=None):
    # data is n samples x m parameters
    linkageMat = scipy.cluster.hierarchy.linkage(data,method=method,metric=metric)
    if nClusters is None:
        clustId = None
    else:
        clustId = scipy.cluster.hierarchy.fcluster(linkageMat,nClusters,'maxclust')
    if plot:
        plt.figure(facecolor='w')
        ax = plt.subplot(1,1,1)
        colorThresh = 0 if nClusters<2 else linkageMat[::-1,2][nClusters-2]
        if colors is not None:
            scipy.cluster.hierarchy.set_link_color_palette(list(colors))
        if labels=='off':
            labels=None
            noLabels=True
        else:
            noLabels=False
        scipy.cluster.hierarchy.dendrogram(linkageMat,ax=ax,color_threshold=colorThresh,above_threshold_color='k',labels=labels,no_labels=noLabels)
        scipy.cluster.hierarchy.set_link_color_palette(None)
        ax.set_yticks([])
        for side in ('right','top','left','bottom'):
            ax.spines[side].set_visible(False)
        if title is not None:
            ax.set_title(title)
        plt.tight_layout()
            
        plt.figure(facecolor='w')
        ax = plt.subplot(1,1,1)
        k = np.arange(linkageMat.shape[0])+2
        if nreps>0:
            randLinkage = np.zeros((nreps,linkageMat.shape[0]))
            shuffledData = data.copy()
            for i in range(nreps):
                for j in range(data.shape[1]):
                    shuffledData[:,j] = data[np.random.permutation(data.shape[0]),j]
                _,m = cluster(shuffledData,method=method,metric=metric)
                randLinkage[i] = m[::-1,2]
            ax.plot(k,np.percentile(randLinkage,2.5,axis=0),'k--')
            ax.plot(k,np.percentile(randLinkage,97.5,axis=0),'k--')
        ax.plot(k,linkageMat[::-1,2],'ko-',mfc='none',ms=10,mew=2,linewidth=2)
        if xmax is None:
            ax.set_xlim([0,k[-1]+1])
        else:
            ax.set_xlim([0,xmax])
        ax.set_xlabel('Cluster')
        ax.set_ylabel('Linkage Distance')
        for side in ('right','top'):
            ax.spines[side].set_visible(False)
        ax.tick_params(direction='out',top=False,right=False)
        if title is not None:
            ax.set_title(title)
        plt.tight_layout()
    
    return clustId,linkageMat

