# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 14:39:37 2023

@author: svc_ccg
"""

import os
import pathlib
import math
import random
import numpy as np
import pandas as pd
import scipy.stats
import scipy.cluster
import scipy.optimize
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


def getUnitsInCluster(units,clustUnitIds,clustIds,clust): 
    return np.in1d(units.index,clustUnitIds[np.in1d(clustUnitIds,units.index) & (clustIds==clust)])


def getUnitsInRegion(units,region,layer=None,rs=False,fs=False):
    if region in ('SC/MRN cluster 1','SC/MRN cluster 2'):
        dirPath = pathlib.Path('//allen/programs/mindscope/workgroups/np-behavior/VBN_video_analysis')
        clustUnitIds = np.load(os.path.join(dirPath,'sc_mrn_clusterUnitId.npy'))
        clustIds = np.load(os.path.join(dirPath,'sc_mrn_clusterId.npy'))
        clust = 1 if '1' in region else 2
        inRegion = getUnitsInCluster(units,clustUnitIds,clustIds,clust)
    else:
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
            if layer is not None:
                inRegion = inRegion & np.in1d(units['cortical_layer'],layer)
            if rs or fs:
                rsUnits = np.array(units['waveform_duration'] > 0.4)
                if rs and not fs:
                    inRegion = inRegion & rsUnits
                elif fs and not rs:
                    inRegion = inRegion & ~rsUnits
        if 'cluster' in region:
            clustTable = pd.read_csv(pathlib.Path('//allen/programs/mindscope/workgroups/np-behavior/vbn_data_release/supplemental_tables/units_with_fast_slow_cluster_ids.csv'))
            clustUnitIds = np.array(clustTable['unit_id'])
            clustIds = np.array(clustTable['fast_slow_cluster_id'])
            clust = 1 if 'cluster 1' in region else 2
            inRegion = inRegion & getUnitsInCluster(units,clustUnitIds,clustIds,clust)
    return inRegion


def apply_unit_quality_filter(unit_table):
 
    qc_filter = [(unit_table['isi_violations']<0.5)&
                (unit_table['amplitude_cutoff']<0.1)&
                (unit_table['presence_ratio']>0.9)&
                (unit_table['quality']=='good')]
   
    return qc_filter[0].values
    

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


def getUnitSamples(sampleSize,nUnits,maxSamples=None):
    if sampleSize>1:
        if sampleSize==nUnits:
            nSamples = 1
            unitSamples = [np.arange(nUnits)]
        else:
            # >99% chance each neuron is chosen at least once
            nSamples = int(math.ceil(math.log(0.01)/math.log(1-sampleSize/nUnits)))
            if maxSamples is not None and nSamples > maxSamples:
                nSamples = maxSamples
            unitSamples = [np.random.choice(nUnits,sampleSize,replace=False) for _ in range(nSamples)]
    else:
        nSamples = nUnits
        unitSamples = [[i] for i in range(nUnits)]
    return unitSamples


def getTrainTestSplits(y,nSplits,hasClasses=True):
    # cross validation using stratified shuffle split
    # each split preserves the percentage of samples of each class
    # all samples used in one test set
    # NaNs in y are place holders to preserve indexing and otherwise ignored

    notNan = ~np.isnan(y)
    if hasClasses:
        classVals = np.unique(y[notNan])
        nClasses = len(classVals)
        nSamples = notNan.sum()
        samplesPerClass = [np.sum(y==val) for val in classVals]
        if any(n < nSplits for n in samplesPerClass):
            return None,None
    else:
        classVals = [None]
        samplesPerClass = [notNan.sum()] 
    samplesPerSplit = [round(n/nSplits) for n in samplesPerClass]
    shuffleInd = np.random.permutation(np.where(notNan)[0])
    trainInd = []
    testInd = []
    for k in range(nSplits):
        testInd.append([])
        for val,n in zip(classVals,samplesPerSplit):
            start = k*n
            ind = shuffleInd[y[shuffleInd]==val] if hasClasses else shuffleInd
            testInd[-1].extend(ind[start:start+n] if k+1<nSplits else ind[start:])
        trainInd.append(np.setdiff1d(shuffleInd,testInd[-1]))
    return trainInd,testInd


def trainDecoder(model,X,y,nSplits):
    classVals = np.unique(y)
    nClasses = len(classVals)
    nSamples = len(y)
    cv = {'estimator': [sklearn.base.clone(model) for _ in range(nSplits)]}
    cv['train_score'] = []
    cv['test_score'] = []
    cv['predict'] = np.full(nSamples,np.nan)
    cv['predict_proba'] = np.full((nSamples,nClasses),np.nan)
    cv['decision_function'] = np.full((nSamples,nClasses),np.nan) if nClasses>2 else np.full(nSamples,np.nan)
    cv['feature_importance'] = []
    cv['coef'] = []
    modelMethods = dir(model)
    trainInd,testInd = getTrainTestSplits(y,nSplits)
    for estimator,train,test in zip(cv['estimator'],trainInd,testInd):
        estimator.fit(X[train],y[train])
        cv['train_score'].append(estimator.score(X[train],y[train]))
        cv['test_score'].append(estimator.score(X[test],y[test]))
        cv['predict'][test] = estimator.predict(X[test])
        for method in ('predict_proba','decision_function'):
            if method in modelMethods:
                cv[method][test] = getattr(estimator,method)(X[test])
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


def fitAccumulatorBrute(accumulatorInput,y,lickLatency,thresholdRange,leakRange,tauARange,tauIRange,alphaIRange,nonDecisionTimeRange):
    # accuracy = np.full((len(thresholdRange),len(leakRange),len(tauARange),len(tauIRange),len(alphaIRange),len(nonDecisionTimeRange)),np.nan)
    # respTime = accuracy.copy()
    totalError = np.full((len(thresholdRange),len(leakRange),len(tauARange),len(tauIRange),len(alphaIRange),len(nonDecisionTimeRange)),np.nan)
    for i,thresh in enumerate(thresholdRange):
        for j,leak in enumerate(leakRange):
            for k,tauA in enumerate(tauARange):
                for l,tauI in enumerate(tauIRange):
                    for m,alphaI in enumerate(alphaIRange):
                        for n,nonDecisionTime in enumerate(nonDecisionTimeRange):
                            resp,rt = runAccumulator(accumulatorInput,thresh,leak,tauA,tauI,alphaI)[:2]
                            # accuracy[i,j] = sklearn.metrics.accuracy_score(y,resp)
                            # respTime[i,j] = np.nanmean(rt)
                            respError = abs(np.mean(resp) - np.mean(y)) / np.mean(y)
                            if not np.any(resp):
                                respTimeError = 1
                            else:
                                meanLat = np.nanmean(lickLatency) * 1000
                                respTimeError = abs((np.nanmean(rt) + nonDecisionTime) - meanLat) / meanLat 
                            totalError[i,j,k,l,m,n] = respError + respTimeError          
    # i,j = np.unravel_index(np.argmax(accuracy),accuracy.shape)
    # leakFit = leakRange[j]
    # thresholdFit = thresholdRange[i]
    i,j,k,l,m,n = np.where(totalError==totalError.min())
    s = np.stack((i,j,k,l,m,n))
    s = np.argmin(np.sum((s - np.mean(s,axis=1)[:,None])**2,axis=0)**0.5)
    thresholdFit = thresholdRange[i[s]]
    leakFit = leakRange[j[s]]
    tauAFit = tauARange[k[s]]
    tauIFit = tauIRange[l[s]]
    alphaIFit = alphaIRange[m[s]]
    nonDecisionTimeFit = nonDecisionTimeRange[n[s]]
    return thresholdFit,leakFit,tauAFit,tauIFit,alphaIFit,nonDecisionTimeFit


def fitAccumulator(accumulatorInput,y,lickLatency,thresholdRange,leakRange,tauARange,tauIRange,alphaIRange,sigmaRange,nonDecisionTimeRange):
    bounds = (thresholdRange,leakRange,tauARange,tauIRange,alphaIRange,sigmaRange,nonDecisionTimeRange)
    fit = scipy.optimize.direct(evalAccumulator,bounds,args=(accumulatorInput,y,lickLatency))
    params = fit.x
    logLoss = fit.fun
    return params,logLoss


def evalAccumulator(params,*args):
    accumulatorInput,y,lickLatency = args
    resp,respTime = runAccumulator(accumulatorInput,*params[:-1])[:2]
    # logLoss = sklearn.metrics.log_loss(y,resp)
    respError = abs(np.mean(resp) - np.mean(y)) / np.mean(y)
    if not np.any(resp):
        respTimeError = 1
    else:
        nonDecisionTime = params[-1]
        m = np.nanmean(lickLatency) * 1000
        respTimeError = abs((np.nanmean(respTime) + nonDecisionTime) - m) / m
    return respError + respTimeError


def runAccumulator(accumulatorInput,threshold,leak,tauA=1,tauI=0,alphaI=0,sigma=0,nReps=10,recordValues=False):
    nReps = nReps if sigma > 0 else 1
    nTrials = len(accumulatorInput)
    resp = np.zeros((nReps,nTrials),dtype=bool)
    respTime = np.full((nReps,nTrials),np.nan)
    accumulatorValue = []
    for n in range(nReps):
        if recordValues:
            accumulatorValue.append([])
        for trial,trialInput in enumerate(accumulatorInput):
            if recordValues:
                accumulatorValue[-1].append([])
            a = 0
            i = 0
            for t,s in enumerate(trialInput):
                sn = s / (alphaI + i) if i > 0 else s
                a += (sn + random.gauss(0,sigma) - a*leak) / tauA
                if tauI > 0:
                    i += (s - i) / tauI
                if not resp[n,trial] and a > threshold:
                    resp[n,trial] = True
                    respTime[n,trial] = t
                    if not recordValues:
                        break
                if recordValues:
                    accumlatorValue[-1][-1].append(a)
    resp = np.mean(resp,axis=0)
    respTime = np.nanmean(respTime,axis=0)
    return resp,respTime,accumulatorValue

