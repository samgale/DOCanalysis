# -*- coding: utf-8 -*-
"""
Created on Fri Sep 29 14:39:37 2023

@author: svc_ccg
"""

import math
import numpy as np
import scipy.stats
import sklearn


def findNearest(array,values):
    ind = np.searchsorted(array,values,side='left')
    for i,j in enumerate(ind):
        if j > 0 and (j == len(array) or math.fabs(values[i] - array[j-1]) < math.fabs(values[i] - array[j])):
            ind[i] = j-1
    return ind


def getFlashTimes(stim):
    # stim = stimulus table or index of
    flashTimes = np.array(stim['start_time'])
    changeTimes = flashTimes[stim['is_change']]
    hit = np.array(stim['hit'][stim['is_change']])
    engaged = np.array([np.sum(hit[(changeTimes>t-60) & (changeTimes<t+60)]) > 1 for t in flashTimes])
    autoRewarded = np.array(stim['auto_rewarded']).astype(bool)
    changeFlashes = np.array(stim['is_change'] & ~autoRewarded & engaged)
    nonChangeFlashes = np.array(engaged &
                                (~stim['is_change']) & 
                                (~stim['omitted']) & 
                                (~stim['previous_omitted']) & 
                                (stim['flashes_since_change']>5) &
                                (stim['flashes_since_last_lick']>1))
    lick = np.array(stim['lick_for_flash'])
    lickTimes = np.array(stim['lick_time'])
    lickLatency = lickTimes - flashTimes
    earlyLick = lickLatency < 0.15
    lateLick = lickLatency > 0.75
    nonChangeFlashes[earlyLick | lateLick] = False
    return flashTimes, changeFlashes, nonChangeFlashes, lick


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


def crossValidate(model,X,y,nSplits):
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

