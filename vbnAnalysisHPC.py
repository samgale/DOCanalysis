# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 14:37:16 2022

@author: svc_ccg
"""

import argparse
import math
import pathlib
import os
import warnings
import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats
import h5py
import sklearn
from sklearn.svm import LinearSVC
import facemap.process
from vbnAnalysisUtils import dictToHdf5, findNearest, getBehavData, getUnitsInCluster, getUnitsInRegion, apply_unit_quality_filter, findResponsiveUnits, getUnitSamples, getTrainTestSplits, trainDecoder, fitAccumulatorBrute, fitAccumulator, runAccumulator


baseDir = pathlib.Path('//allen/programs/mindscope/workgroups/np-behavior/vbn_data_release/supplemental_tables')

outputDir = pathlib.Path('//allen/programs/mindscope/workgroups/np-behavior/VBN_video_analysis')

stimTable = pd.read_csv(os.path.join(baseDir,'master_stim_table.csv'))

videoTable = pd.read_excel(os.path.join(baseDir,'vbn_video_paths_full_validation.xlsx'))
videoTable.insert(0,'session_id',[int(s[:s.find('_')]) for s in videoTable['exp_id']])

unitTable = pd.read_csv(os.path.join(baseDir,'master_unit_table.csv'))
unitData = h5py.File(os.path.join(baseDir,'vbnAllUnitSpikeTensor.hdf5'),mode='r')

clusterTable = pd.read_csv(os.path.join(baseDir,'unit_cluster_labels.csv'))


def runFacemap(sessionId):
    dlcLabel = 'nose_tip'
    roiParams = {'side': {'xoffset': -3, 'yoffset': -10, 'width': 35, 'height': 35},
                 'face': {'xoffset': -30, 'yoffset': -50, 'width': 70, 'height': 70}}
    
    for videoType,sbin in zip(('side',),(1,)): # ('side','face'),(1,2)
        with h5py.File(os.path.join(baseDir,'dlcData.hdf5')) as dlcData:
            likelihood = dlcData[str(sessionId)][videoType][dlcLabel]['likelihood'][()]
            x,y = [int(np.average(dlcData[str(sessionId)][videoType][dlcLabel][c][()],weights=likelihood)) for c in ('x','y')]
        roi = np.array([x + roiParams[videoType]['xoffset'], y + roiParams[videoType]['yoffset'],
                        roiParams[videoType]['width'], roiParams[videoType]['height']])
        
        videoIndex = np.where(videoTable['session_id'] == sessionId)[0][0]
        videoPath = videoTable.loc[videoIndex,videoType+'_video'].replace('\\','/')
        
        facemapSavePath = os.path.join(outputDir,'facemapData')
        proc = {'sx': np.array([0]),
                'sy': np.array([0]),
                'sbin': sbin,
                'fullSVD': False, # computes SVD for full video and roi if True else just roi
                'save_mat': False, # save .mat file if True
                'rois': [{'rind': 1,
                          'rtype': 'motion SVD',
                          'ivid': 0,
                          'xrange': np.arange(roi[0],roi[0]+roi[2]),
                          'yrange': np.arange(roi[1],roi[1]+roi[3])}],
                'savepath': facemapSavePath}
        
        facemap.process.run(filenames=[[videoPath]],
                            sbin=sbin,
                            motSVD=True,
                            movSVD=True, # throws an indexing error if False
                            GUIobject=None,
                            parent=None,
                            proc=proc,
                            savepath=facemapSavePath)
        
        videoName = os.path.basename(videoPath)
        facemapDataPath = os.path.join(facemapSavePath,videoName[:-4]+'_proc.npy')
        facemapData = np.load(facemapDataPath,allow_pickle=True).item()
        
        h5Out = h5py.File(os.path.join(outputDir,facemapDataPath[:-3]+'hdf5'),'w')
        d = {}
        d['xrange'] = facemapData['rois'][0]['xrange']
        d['yrange'] = facemapData['rois'][0]['yrange']
        d['xrange_bin'] = facemapData['rois'][0]['xrange_bin']
        d['yrange_bin'] = facemapData['rois'][0]['yrange_bin']
        d['avgframe'] = facemapData['avgframe_reshape']
        d['avgmotion'] = facemapData['avgmotion_reshape']
        d['motSv'] = facemapData['motSv']
        d['motSVD'] = facemapData['motSVD'][1]
        d['motMask'] = facemapData['motMask_reshape'][1]
        d['motion'] = facemapData['motion'][1]
        for key,val in d.items():
            h5Out.create_dataset(key,data=val,compression='gzip',compression_opts=4)
        h5Out.close()
        
        os.remove(facemapDataPath)


def decodeFromFacemap(sessionId):
    model = LinearSVC(C=1.0,max_iter=int(1e4),class_weight=None)
    nCrossVal = 5
    decodeWindowStart = 0
    decodeWindowEnd = 0.75
    frameInterval = 1/60
    decodeWindows = np.arange(decodeWindowStart,decodeWindowEnd+frameInterval/2,frameInterval)
    
    stim = stimTable[(stimTable['session_id']==sessionId) & stimTable['active']].reset_index()
    flashTimes,changeFlashes,catchFlashes,nonChangeFlashes,omittedFlashes,prevOmittedFlashes,novelFlashes,lick,lickTimes = getBehavData(stim)
    
    flashSvd = []
    videoIndex = np.where(videoTable['session_id'] == sessionId)[0][0]
    for videoType in ('side',): # ('side','face')
        videoPath = videoTable.loc[videoIndex,videoType+'_video'].replace('\\','/')
        videoName = os.path.basename(videoPath)
        facemapDataPath = os.path.join(outputDir,'facemapData',videoName[:-4]+'_proc.hdf5')
        with h5py.File(facemapDataPath) as facemapData:
            svd = facemapData['motSVD'][()]
        frameTimesPath = videoTable.loc[videoIndex,videoType+'_timestamp_path'].replace('\\','/')
        frameTimes = np.load(frameTimesPath)
        flashSvd.append([])
        for flashTime in flashTimes:
            frameIndex = findNearest(frameTimes,decodeWindows+flashTime)
            flashSvd[-1].append(svd[frameIndex])
    flashSvd = np.concatenate(flashSvd,axis=2)
    
    decoderLabels = ('non-change lick','change lick novel','change no lick novel','non-change lick novel','non-change no lick novel')
    d = {metric: {lbl: [] for lbl in decoderLabels} for metric in ('trainAccuracy','featureWeights','accuracy','balancedAccuracy','prediction','confidence')}
    d['decodeWindows'] = decodeWindows
    d['changeFlashes'] = changeFlashes
    d['nonChangeFlashes'] = nonChangeFlashes
    d['novelFlashes'] = novelFlashes
    d['lick'] = lick
    
    for lbl,flashes,ind in zip(decoderLabels,
                               (nonChangeFlashes,changeFlashes & lick,changeFlashes & ~lick,nonChangeFlashes & lick,nonChangeFlashes & ~lick),
                               (lick,novelFlashes,novelFlashes,novelFlashes,novelFlashes)):
        y = ind[flashes]
        if np.sum(y) >= 10 and np.sum(~y) >= 10:
            warnings.filterwarnings('ignore')
            for i in range(len(decodeWindows)):
                X = flashSvd[flashes,:i+1].reshape(flashes.sum(),-1)   
                cv = trainDecoder(model,X,y,nCrossVal)
                d['trainAccuracy'][lbl].append(np.mean(cv['train_score']))
                d['featureWeights'][lbl].append(np.mean(cv['coef'],axis=0).squeeze())
                d['accuracy'][lbl].append(np.mean(cv['test_score']))
                d['balancedAccuracy'][lbl].append(sklearn.metrics.balanced_accuracy_score(y,cv['predict']))
                d['prediction'][lbl].append(cv['predict'])
                d['confidence'][lbl].append(cv['decision_function'])
            warnings.filterwarnings('default')
    
    np.save(os.path.join(outputDir,'facemapDecoding','facemapDecoding_'+str(sessionId)+'.npy'),d)


def decodeLicksFromUnits(sessionId):
    useResponsiveUnits = False
    model = LinearSVC(C=1.0,max_iter=int(1e4),class_weight=None)
    nCrossVal = 5
    unitSampleSize = [1,5,10,15,20,25,30,40,50,60]
    decodeWindowSize = 10
    decodeWindowEnd = 750
    decodeWindows = np.arange(decodeWindowSize,decodeWindowEnd+decodeWindowSize,decodeWindowSize)

    unitSampleDecodeWindow = 750
    decodeWindowSampleSize = 20

    regions = ('LGd','VISp','VISl','VISrl','VISal','VISpm','VISam','LP',
               'MRN','MB','SC','APN','NOT','Hipp') #+ ('all',)

    clusters = ['all'] #+ ['cluster '+str(c) for c in np.unique(clusterTable['cluster_labels']) + 1]

    baseWin = slice(680,750)
    respWin = slice(30,100)

    units = unitTable.set_index('unit_id').loc[unitData[str(sessionId)]['unitIds'][:]]
    spikes = unitData[str(sessionId)]['spikes']

    qualityUnits = apply_unit_quality_filter(units)
    
    stim = stimTable[(stimTable['session_id']==sessionId) & stimTable['active']].reset_index()
    flashTimes,changeFlashes,catchFlashes,nonChangeFlashes,omittedFlashes,prevOmittedFlashes,novelFlashes,lick,lickTimes = getBehavData(stim)
    
    d = {region: {cluster: {sampleSize: {metric: [] for metric in ('trainAccuracy','featureWeights','accuracy','balancedAccuracy','prediction','confidence')}
         for sampleSize in unitSampleSize} for cluster in clusters} for region in regions}
    d['regions'] = regions
    d['clusters'] = clusters
    d['unitSampleSize'] = unitSampleSize
    d['decodeWindows'] = decodeWindows
    d['unitSampleDecodeWindow'] = unitSampleDecodeWindow
    d['decodeWindowSampleSize'] = decodeWindowSampleSize
    d['lick'] = lick[nonChangeFlashes]

    y = lick[nonChangeFlashes]

    warnings.filterwarnings('ignore')
    for region in regions:
        inRegion = qualityUnits if region=='all' else qualityUnits & getUnitsInRegion(units,region)
        for clustNum,cluster in enumerate(clusters):
            unitsToUse = inRegion if cluster=='all' else inRegion & getUnitsInCluster(units,clusterTable['unit_id'],clusterTable['cluster_labels'],clustNum-1)
            nUnits = unitsToUse.sum()
            if nUnits < 1:
                continue
                
            sp = np.zeros((nUnits,spikes.shape[1],spikes.shape[2]),dtype=bool)
            for i,u in enumerate(np.where(unitsToUse)[0]):
                sp[i]=spikes[u,:,:]
            
            if useResponsiveUnits:
                changeSp = sp[:,changeFlashes,:]
                preChangeSp = sp[:,np.where(changeFlashes)[0]-1,:]
                hasResp = findResponsiveUnits(preChangeSp,changeSp,baseWin,respWin)
                sp = sp[hasResp]
                nUnits = hasResp.sum()

            flashSp = sp[:,nonChangeFlashes,:decodeWindows[-1]].reshape((nUnits,nonChangeFlashes.sum(),len(decodeWindows),decodeWindowSize)).sum(axis=-1)
            
            for sampleSize in unitSampleSize:
                if nUnits < sampleSize:
                    continue
                unitSamples = getUnitSamples(sampleSize,nUnits)

                for winEnd in decodeWindows:
                    if sampleSize!=decodeWindowSampleSize and winEnd!=unitSampleDecodeWindow:
                        continue
                    winEnd = int(winEnd/decodeWindowSize)
                    for metric in d[region][cluster][sampleSize]:
                        d[region][cluster][sampleSize][metric].append([])
                    for unitSamp in unitSamples:
                        X = flashSp[unitSamp,:,:winEnd].transpose(1,0,2).reshape((nonChangeFlashes.sum(),-1))                        
                        cv = trainDecoder(model,X,y,nCrossVal)
                        d[region][cluster][sampleSize]['trainAccuracy'][-1].append(np.mean(cv['train_score']))
                        d[region][cluster][sampleSize]['featureWeights'][-1].append(np.mean(cv['coef'],axis=0).squeeze())
                        d[region][cluster][sampleSize]['accuracy'][-1].append(np.mean(cv['test_score']))
                        d[region][cluster][sampleSize]['balancedAccuracy'][-1].append(sklearn.metrics.balanced_accuracy_score(y,cv['predict']))
                        d[region][cluster][sampleSize]['prediction'][-1].append(cv['predict'])
                        d[region][cluster][sampleSize]['confidence'][-1].append(cv['decision_function'])
                    for metric in d[region][cluster][sampleSize]:
                        if metric == 'prediction':
                            d[region][cluster][sampleSize][metric][-1] = np.mean(d[region][cluster][sampleSize][metric][-1],axis=0)
                        else:
                            d[region][cluster][sampleSize][metric][-1] = np.median(d[region][cluster][sampleSize][metric][-1],axis=0)
    warnings.filterwarnings('default')

    np.save(os.path.join(outputDir,'unitLickDecoding','unitLickDecoding_'+str(sessionId)+'.npy'),d)


def decodeChange(sessionId):
    useResponsiveUnits = False
    trainOnFlashesWithLicks = True

    nCrossVal = 5
    unitSampleSize = [1,5,10,15,20,25,30,40,50,60]
    decodeWindowSize = 10
    decodeWindowEnd = 750
    decodeWindows = np.arange(decodeWindowSize,decodeWindowEnd+decodeWindowSize,decodeWindowSize)

    unitSampleDecodeWindow = 750
    decodeWindowSampleSize = 20
   
    regions = ('LGd','VISp','VISl','VISrl','VISal','VISpm','VISam','LP',
               'MRN','MB','SC','APN','NOT','Hipp') #+ ('all',)

    clusters = ['all'] #+ ['cluster '+str(c) for c in np.unique(clusterTable['cluster_labels']) + 1]

    baseWin = slice(680,750)
    respWin = slice(30,100)

    units = unitTable.set_index('unit_id').loc[unitData[str(sessionId)]['unitIds'][:]]
    spikes = unitData[str(sessionId)]['spikes']

    qualityUnits = apply_unit_quality_filter(units)
    
    stim = stimTable[(stimTable['session_id']==sessionId) & stimTable['active']].reset_index()
    flashTimes,changeFlashes,catchFlashes,nonChangeFlashes,omittedFlashes,prevOmittedFlashes,novelFlashes,lick,lickTimes = getBehavData(stim)
    preChangeFlashes = np.concatenate((changeFlashes[1:],[False]))
    nFlash = len(flashTimes)
    nChange = changeFlashes.sum()
    lickLatency = lickTimes - flashTimes
    imageName = np.array(stim['image_name'])

    outcome = []
    for lbl in ('hit','miss','false_alarm','correct_reject'):
        a = stim[lbl].copy()
        a[a.isnull()] = False
        outcome.append(np.array(a).astype(bool))
    hit,miss,falseAlarm,correctReject = outcome
    hit = hit & changeFlashes
    miss = miss & changeFlashes
    falseAlarm = falseAlarm & catchFlashes
    correctReject = correctReject & catchFlashes

    d = {region: {cluster: {sampleSize: {metric: [] for metric in ('trainAccuracy','featureWeights','accuracy','accuracyFamiliar','accuracyNovel','prediction','confidence')}
         for sampleSize in unitSampleSize} for cluster in clusters} for region in regions}
    d['regions'] = regions
    d['clusters'] = clusters
    d['unitSampleSize'] = unitSampleSize
    d['decodeWindows'] = decodeWindows
    d['unitSampleDecodeWindow'] = unitSampleDecodeWindow
    d['decodeWindowSampleSize'] = decodeWindowSampleSize
    d['imageName'] = imageName
    d['change'] = changeFlashes
    d['preChange'] = preChangeFlashes
    d['catch'] = catchFlashes
    d['nonChange'] = nonChangeFlashes
    d['omitted'] = omittedFlashes
    d['prevOmitted'] = prevOmittedFlashes
    d['novel'] = novelFlashes
    d['lick'] = lick
    d['lickLatency'] = lickLatency
    d['hit'] = hit
    d['miss'] = miss
    d['falseAlarm'] = falseAlarm
    d['correctReject'] = correctReject

    y = np.full(nFlash,np.nan)
    if trainOnFlashesWithLicks:
        y[changeFlashes & lick] = 1
        y[nonChangeFlashes & lick] = 0
    else:
        y[changeFlashes] = 1
        y[preChangeFlashes] = 0

    # get train/test sets
    if np.any(novelFlashes):
        # balance novel and familiar flashes
        train = []
        test = []
        for i in (novelFlashes,~novelFlashes):
            k = y.copy()
            k[i] = np.nan
            tr,ts = getTrainTestSplits(k,nCrossVal)
            train.append(tr)
            test.append(ts)
        trainInd = [list(i)+list(j) for i,j in zip(*train)]
        testInd = [list(i)+list(j) for i,j in zip(*test)]
    else:
        trainInd,testInd = getTrainTestSplits(y,nCrossVal)
    # use all train/test flashes as training set for all other flashes
    trainInd.append(np.where(~np.isnan(y))[0])
    testInd.append(np.where(np.isnan(y))[0])
    
    warnings.filterwarnings('ignore')
    for region in regions:
        inRegion = qualityUnits if region=='all' else qualityUnits & getUnitsInRegion(units,region)
        for clustNum,cluster in enumerate(clusters):
            unitsToUse = inRegion if cluster=='all' else inRegion & getUnitsInCluster(units,clusterTable['unit_id'],clusterTable['cluster_labels'],clustNum-1)
            nUnits = unitsToUse.sum()
            if nUnits < 1:
                continue
                
            sp = np.zeros((nUnits,spikes.shape[1],spikes.shape[2]),dtype=bool)
            for i,u in enumerate(np.where(unitsToUse)[0]):
                sp[i]=spikes[u,:,:]
            
            if useResponsiveUnits:
                changeSp = sp[:,changeFlashes,:]
                preChangeSp = sp[:,np.where(changeFlashes)[0]-1,:]
                hasResp = findResponsiveUnits(preChangeSp,changeSp,baseWin,respWin)
                sp = sp[hasResp]
                nUnits = hasResp.sum()

            flashSp = sp[:,:,:decodeWindows[-1]].reshape((nUnits,nFlash,len(decodeWindows),decodeWindowSize)).sum(axis=-1)
            
            for sampleSize in unitSampleSize:
                if nUnits < sampleSize:
                    continue
                unitSamples = getUnitSamples(sampleSize,nUnits)

                for winEnd in decodeWindows:
                    if sampleSize!=decodeWindowSampleSize and winEnd!=unitSampleDecodeWindow:
                        continue
                    winEnd = int(winEnd/decodeWindowSize)

                    for metric in d[region][cluster][sampleSize]:
                        d[region][cluster][sampleSize][metric].append([])

                    for unitSamp in unitSamples:
                        X = flashSp[unitSamp,:,:winEnd].transpose(1,0,2).reshape((nFlash,-1))       
                        trainAccuracy = np.zeros(nCrossVal+1)
                        featureWeights = np.zeros((nCrossVal+1,X.shape[1]))
                        accuracy = np.zeros(nCrossVal)
                        accuracyFamiliar = np.zeros(nCrossVal)
                        accuracyNovel = np.zeros(nCrossVal)
                        prediction = np.zeros(nFlash)
                        confidence = np.zeros(nFlash)
                        for k,(train,test) in enumerate(zip(trainInd,testInd)):
                            decoder = LinearSVC(C=1.0,max_iter=int(1e4),class_weight=None)
                            decoder.fit(X[train],y[train])
                            trainAccuracy[k] = sklearn.metrics.balanced_accuracy_score(y[train],decoder.predict(X[train]))
                            featureWeights[k] = np.squeeze(decoder.coef_)
                            if k < nCrossVal:
                                accuracy[k] = sklearn.metrics.balanced_accuracy_score(y[test],decoder.predict(X[test]))
                                if np.any(novelFlashes):
                                    accuracyFamiliar[k] = sklearn.metrics.balanced_accuracy_score(y[test][~novelFlashes[test]],decoder.predict(X[test][~novelFlashes[test]]))
                                    accuracyNovel[k] = sklearn.metrics.balanced_accuracy_score(y[test][novelFlashes[test]],decoder.predict(X[test][novelFlashes[test]]))
                            prediction[test] = decoder.predict(X[test])
                            confidence[test] = decoder.decision_function(X[test])
                        d[region][cluster][sampleSize]['trainAccuracy'][-1].append(trainAccuracy[-1])
                        d[region][cluster][sampleSize]['featureWeights'][-1].append(featureWeights[-1])
                        d[region][cluster][sampleSize]['accuracy'][-1].append(np.mean(accuracy))
                        d[region][cluster][sampleSize]['accuracyFamiliar'][-1].append(np.mean(accuracyFamiliar))
                        d[region][cluster][sampleSize]['accuracyNovel'][-1].append(np.mean(accuracyNovel))
                        d[region][cluster][sampleSize]['prediction'][-1].append(prediction)
                        d[region][cluster][sampleSize]['confidence'][-1].append(confidence)

                    # take median of decoding metrics across unit samples 
                    d[region][cluster][sampleSize]['trainAccuracy'][-1] = np.median(d[region][cluster][sampleSize]['trainAccuracy'][-1])
                    d[region][cluster][sampleSize]['featureWeights'][-1] = np.median(d[region][cluster][sampleSize]['featureWeights'][-1],axis=(0,1))
                    d[region][cluster][sampleSize]['accuracy'][-1] = np.median(d[region][cluster][sampleSize]['accuracy'][-1],axis=0)
                    d[region][cluster][sampleSize]['accuracyFamiliar'][-1] = np.median(d[region][cluster][sampleSize]['accuracyFamiliar'][-1],axis=0)
                    d[region][cluster][sampleSize]['accuracyNovel'][-1] = np.median(d[region][cluster][sampleSize]['accuracyNovel'][-1],axis=0)
                    d[region][cluster][sampleSize]['prediction'][-1] = np.mean(d[region][cluster][sampleSize]['prediction'][-1],axis=0)
                    d[region][cluster][sampleSize]['confidence'][-1] = np.median(d[region][cluster][sampleSize]['confidence'][-1],axis=0)
    warnings.filterwarnings('default')

    np.save(os.path.join(outputDir,'unitChangeDecoding','unitChangeDecoding_'+str(sessionId)+'.npy'),d)


def pooledDecoding(label,region,cluster):
    decodeWindowSize = 10
    decodeWindowEnd = 750
    decodeWindows = np.arange(decodeWindowSize,decodeWindowEnd+decodeWindowSize,decodeWindowSize)
    minFlashes = 20
    unitSampleSize = 20
    nUnitSamples = 100
    nPseudoFlashes = 100

    flashSpikes = [] # binned spikes for all flashes of each label for each session with neurons in region and cluster
    unitIndex = [] # nUnits x (session, unit in session)
    sessionIndex = 0
    for sessionId in stimTable['session_id'].unique():
        stim = stimTable[(stimTable['session_id']==sessionId) & stimTable['active']].reset_index()
        flashTimes,changeFlashes,catchFlashes,nonChangeFlashes,omittedFlashes,prevOmittedFlashes,novelFlashes,lick,lickTimes = getBehavData(stim)
        imageName = np.array(stim['image_name'])
        if label == 'change':
            flashInd = (nonChangeFlashes & lick, changeFlashes & lick)
        elif label == 'lick':
            flashInd = (nonChangeFlashes & ~lick, nonChangeFlashes & lick)
        elif label == 'hit':
            flashInd = (changeFlashes & ~lick, changeFlashes & lick)
        elif label == 'image':   
            flashInd = tuple(nonChangeFlashes & ~lick & (imageName==img) for img in np.unique(imageName) if img != 'omitted')
        elif label == 'visual_response':
            flashInd = (omittedFlashes & ~lick, nonChangeFlashes & ~lick)
        if any(flashes.sum() < minFlashes for flashes in flashInd):
            continue

        units = unitTable.set_index('unit_id').loc[unitData[str(sessionId)]['unitIds'][:]]
        qualityUnits = apply_unit_quality_filter(units)
        inRegion = qualityUnits if region=='all' else qualityUnits & getUnitsInRegion(units,region)
        unitsToUse = inRegion if cluster=='all' else inRegion & getUnitsInCluster(units,clusterTable['unit_id'],clusterTable['cluster_labels'],int(cluster[cluster.find('_')+1:])-1)
        nUnits = unitsToUse.sum()
        if nUnits < 1:
            continue
        spikes = unitData[str(sessionId)]['spikes']     
        sp = np.zeros((nUnits,spikes.shape[1],spikes.shape[2]),dtype=bool)
        for i,u in enumerate(np.where(unitsToUse)[0]):
            sp[i] = spikes[u,:,:]
        
        for f,flashes in enumerate(flashInd):
            if sessionIndex == 0:
                flashSpikes.append([])
            flashSpikes[f].append(sp[:,flashes,:decodeWindows[-1]].reshape((nUnits,flashes.sum(),len(decodeWindows),decodeWindowSize)).sum(axis=-1))
        
        unitIndex.append(np.stack((np.full(nUnits,sessionIndex),np.arange(nUnits))).T)
        sessionIndex += 1
    if sum(len(i) for i in unitIndex) < unitSampleSize:
        return
    unitIndex = np.concatenate(unitIndex)

    unitSamples = [np.random.choice(unitIndex.shape[0],unitSampleSize,replace=False) for _ in range(nUnitSamples)]
    y = np.repeat(np.arange(len(flashInd)),nPseudoFlashes)
    accuracy = np.zeros((nUnitSamples,len(decodeWindows)))
    warnings.filterwarnings('ignore')
    for i,unitSamp in enumerate(unitSamples):
        pseudoTrain = [np.zeros((unitSampleSize,nPseudoFlashes,len(decodeWindows))) for _ in range(len(flashInd))]
        pseudoTest = [np.zeros((unitSampleSize,nPseudoFlashes,len(decodeWindows))) for _ in range(len(flashInd))]
        for f,flashes in enumerate(flashInd):
            for k,(s,u) in enumerate(unitIndex[unitSamp]):
                n = flashSpikes[f][s].shape[1]
                r = np.random.permutation(n)
                train = r[n//2:]
                test = r[:n//2]
                pseudoTrain[f][k] = flashSpikes[f][s][u,np.random.choice(train,nPseudoFlashes,replace=True)]
                pseudoTest[f][k] = flashSpikes[f][s][u,np.random.choice(test,nPseudoFlashes,replace=True)]
        pseudoTrain = np.concatenate(pseudoTrain,axis=1)
        pseudoTest = np.concatenate(pseudoTest,axis=1)
        for j,winEnd in enumerate((decodeWindows/decodeWindowSize).astype(int)):
            Xtrain = pseudoTrain[:,:,:winEnd].transpose(1,0,2).reshape((len(y),-1))
            Xtest = pseudoTest[:,:,:winEnd].transpose(1,0,2).reshape((len(y),-1))
            decoder = LinearSVC(C=1.0,max_iter=int(1e4),class_weight=None)
            decoder.fit(Xtrain,y)
            accuracy[i,j] = decoder.score(Xtest,y) 
    warnings.filterwarnings('default')

    if label == 'change':
        dirName = 'pooledChangeDecoding'
    elif label == 'lick':
        dirName = 'pooledLickDecoding'
    elif label == 'hit':
        dirName = 'pooledHitDecoding'
    elif label == 'image':
        dirName = 'pooledImageDecoding'
    elif label == 'visual_response':
        dirName = 'pooledVisualResponseDecoding'
    np.save(os.path.join(outputDir,dirName,dirName+'_'+region+'_'+cluster+'.npy'),accuracy)


def fitIntegratorModel(sessionId):
    regions = ('VISall',)
    layer = None
    bruteFit = True
    nCrossVal = 5
    nShuffles = 1
    minUnits = 20
    baseWin = slice(680,750)
    respWin = slice(30,100)
    tEnd = 200
    binSize = 1
    nBins = int(tEnd/binSize)
    if bruteFit:
        thresholdRange = np.arange(20,55,5)
        leakRange = np.arange(0.02,0.25,0.02)
        sigmaRange = np.arange(0,10,1)
        nonDecisionTimeRange = np.arange(100,500,50)
        tauARange = np.arange(5,100,10)
        tauIRange = np.arange(5,100,10)
        alphaIRange = np.arange(1,10,1)
    else:
        thresholdRange = (0,100)
        leakRange = (0,1)
        sigmaRange = (0,10)
        nonDecisionTimeRange = (0,750)
        tauARange = (1,200)
        tauIRange = (1,200)
        alphaIRange = (0.1,10)

    units = unitTable.set_index('unit_id').loc[unitData[str(sessionId)]['unitIds'][:]]
    qualityUnits = apply_unit_quality_filter(units)
    spikes = unitData[str(sessionId)]['spikes']
    
    stim = stimTable[(stimTable['session_id']==sessionId) & stimTable['active']].reset_index()
    flashTimes,changeFlashes,catchFlashes,nonChangeFlashes,omittedFlashes,prevOmittedFlashes,novelFlashes,lick,lickTimes = getBehavData(stim)
    preChangeFlashes = np.concatenate((changeFlashes[1:],[False]))
    lickLatency = lickTimes - flashTimes
    nFlash = len(flashTimes)
    nChange = changeFlashes.sum()

    outcome = []
    for lbl in ('hit','miss','false_alarm','correct_reject'):
        a = stim[lbl].copy()
        a[a.isnull()] = False
        outcome.append(np.array(a).astype(bool))
    hit,miss,falseAlarm,correctReject = outcome
    hit = hit & changeFlashes
    miss = miss & changeFlashes
    falseAlarm = falseAlarm & catchFlashes
    correctReject = correctReject & catchFlashes
    if hit.sum() < nCrossVal or miss.sum() < nCrossVal:
        return
    
    d = {region: {} for region in regions}
    d['sessionId'] = sessionId
    d['regions'] = regions
    d['tEnd'] = tEnd
    d['binSize'] = binSize
    d['thresholdRange'] = thresholdRange
    d['leakRange'] = leakRange
    d['sigmaRange'] = sigmaRange
    d['nonDecisionTimeRange'] = nonDecisionTimeRange
    d['tauARange'] = tauARange
    d['tauIRange'] = tauIRange
    d['alphaIRange'] = alphaIRange
    d['imageName'] = np.array(stim['image_name'])
    d['change'] = changeFlashes
    d['preChange'] = preChangeFlashes
    d['catch'] = catchFlashes
    d['nonChange'] = nonChangeFlashes
    d['omitted'] = omittedFlashes
    d['prevOmitted'] = prevOmittedFlashes
    d['novel'] = novelFlashes
    d['lick'] = lick
    d['lickLatency'] = lickLatency
    d['hit'] = hit
    d['miss'] = miss
    d['falseAlarm'] = falseAlarm
    d['correctReject'] = correctReject

    y = {}
    trainInd = {}
    testInd = {}
    for go,nogo,lbl in zip((changeFlashes,hit | falseAlarm,(changeFlashes | catchFlashes | nonChangeFlashes) & lick),
                           (preChangeFlashes,miss | correctReject,(changeFlashes | catchFlashes | nonChangeFlashes) & ~lick),
                           ('change','hit','all')):
        y[lbl] = np.full(nFlash,np.nan)
        y[lbl][go] = 1
        y[lbl][nogo] = 0
        if lbl=='change' and np.any(novelFlashes):
            # balance novel and familiar changes
            train = []
            test = []
            for i in (novelFlashes,~novelFlashes):
                j = np.where(changeFlashes & i)[0]
                j = np.concatenate((j,j-1))
                k = y[lbl].copy()
                k[j] = np.nan
                tr,ts = getTrainTestSplits(k,nCrossVal)
                train.append(tr)
                test.append(ts)
            trainInd[lbl] = [list(i)+list(j) for i,j in zip(*train)]
            testInd[lbl] = [list(i)+list(j) for i,j in zip(*test)]
        else:
            trainInd[lbl],testInd[lbl] = getTrainTestSplits(y[lbl],nCrossVal)
        trainInd[lbl].append(np.where(~np.isnan(y[lbl]))[0])
        testInd[lbl].append(np.where(np.isnan(y[lbl]))[0])

    warnings.filterwarnings('ignore')
    for region in regions:
        unitsToUse = qualityUnits & getUnitsInRegion(units,region,layer=layer,rs=True)
        nUnits = unitsToUse.sum()
        if nUnits < minUnits:
            continue
        d[region]['nUnits'] = nUnits

        sp = np.zeros((nUnits,spikes.shape[1],spikes.shape[2]),dtype=bool)
        for i,u in enumerate(np.where(unitsToUse)[0]):
            sp[i] = spikes[u,:,:]
            
        baseline = sp[:,:,-tEnd:].mean(axis=-1)
        baseline = np.concatenate((baseline[:,0][:,None],baseline[:,:-1]),axis=1)
        sp = sp[:,:,:tEnd] - baseline[:,:,None]
        if binSize > 1:
            sp = sp.reshape((nUnits,nFlash,nBins,binSize)).sum(axis=-1)
        
        flashSp = sp.mean(axis=0)
        inputNorm = 1 # flashSp.max()
        flashSp /= inputNorm
        flashSp *= 1000
        d[region]['inputNorm'] = inputNorm
        d[region]['integratorInput'] = flashSp

        d[region]['threshold'] = {}
        d[region]['leak'] = {}
        d[region]['sigma'] = {}
        d[region]['nonDecisionTime'] = {}
        d[region]['tauA'] = {}
        d[region]['tauI'] = {}
        d[region]['alphaI'] = {}
        d[region]['integratorTrainLogLoss'] = {}
        d[region]['integratorTrainAccuracy'] = {}
        d[region]['integratorTrainRespTime'] = {}
        d[region]['integratorResp'] = {}
        d[region]['integratorRespTime'] = {}
        d[region]['integratorShuffledAccuracy'] = {}
        for lbl in ('hit',):
            thresholdFit = np.full(nCrossVal+1,np.nan)
            leakFit = thresholdFit.copy()
            sigmaFit = thresholdFit.copy()
            nonDecisionTimeFit = thresholdFit.copy()
            tauAFit = thresholdFit.copy()
            tauIFit = thresholdFit.copy()
            alphaIFit = thresholdFit.copy()
            integratorTrainLogLoss = thresholdFit.copy()
            integratorTrainAccuracy = np.full((nCrossVal+1,len(thresholdRange),len(leakRange)),np.nan)
            integratorTrainRespTime = integratorTrainAccuracy.copy()
            integratorResp = np.full(nFlash,np.nan)
            integratorRespTime = integratorResp.copy()
            for k,(train,test) in enumerate(zip(trainInd[lbl],testInd[lbl])):
                if bruteFit:
                    thresholdFit[k],leakFit[k],tauAFit[k],tauIFit[k],alphaIFit[k],nonDecisionTimeFit[k] = fitAccumulatorBrute(flashSp[train],y[lbl][train],lickLatency[train],thresholdRange,leakRange,tauARange,tauIRange,alphaIRange,nonDecisionTimeRange)
                    integratorResp[test],integratorRespTime[test] = runAccumulator(flashSp[test],thresholdFit[k],leakFit[k],tauAFit[k],tauIFit[k],alphaIFit[k])[:2]
                else:
                    params,integratorTrainLogLoss[k] = fitAccumulator(flashSp[train],y[lbl][train],lickLatency[train],thresholdRange,leakRange,tauARange,tauIRange,alphaIRange,sigmaRange,nonDecisionTimeRange)
                    thresholdFit[k],leakFit[k],tauAFit[k],tauIFit[k],alphaIFit[k],sigmaFit[k],nonDecisionTimeFit[k] = params
                    integratorResp[test],integratorRespTime[test] = runAccumulator(flashSp[test],*params[:-1])[:2]
            d[region]['threshold'][lbl] = thresholdFit
            d[region]['leak'][lbl] = leakFit
            d[region]['sigma'][lbl] = sigmaFit
            d[region]['nonDecisionTime'][lbl] = nonDecisionTimeFit
            d[region]['tauA'][lbl] = tauAFit
            d[region]['tauI'][lbl] = tauIFit
            d[region]['alphaI'][lbl] = alphaIFit
            d[region]['integratorTrainLogLoss'][lbl] = integratorTrainLogLoss
            d[region]['integratorTrainAccuracy'][lbl] = integratorTrainAccuracy
            d[region]['integratorTrainRespTime'][lbl] = integratorTrainRespTime
            d[region]['integratorResp'][lbl] = integratorResp
            d[region]['integratorRespTime'][lbl] = integratorRespTime
            # if lbl == 'hit':
            #     d[region]['integratorShuffledAccuracy'][lbl] = []
            #     for _ in range(nShuffles):
            #         resp = np.full(nFlash,np.nan)
            #         for train,test in zip(trainInd[lbl][:-1],testInd[lbl][:-1]):
            #             shuffleInd = np.random.permutation(train)
            #             leak,thresh = fitAccumulator(flashSp[train],y[lbl][shuffleInd],leakRange,thresholdRange)[:2]
            #             resp[test] = runAccumulator(flashSp[test],leak,thresh)[0]
            #         d[region]['integratorShuffledAccuracy'][lbl].append(sklearn.metrics.balanced_accuracy_score(y[lbl][trainInd[lbl][-1]],resp[trainInd[lbl][-1]]))
    warnings.filterwarnings('default')

    h5File = h5py.File(os.path.join(outputDir,'integratorModel','integratorModel_'+str(sessionId)+'.hdf5'),'w')
    dictToHdf5(h5File,d)
    h5File.close()

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # parser.add_argument('--sessionId',type=int)
    # args = parser.parse_args()
    # sessionId = args.sessionId
    # # runFacemap(sessionId)
    # # decodeFromFacemap(sessionId)
    # # decodeLicksFromUnits(sessionId)
    # # decodeChange(sessionId)
    # # fitIntegratorModel(sessionId)

    parser.add_argument('--label',type=str)
    parser.add_argument('--region',type=str)
    parser.add_argument('--cluster',type=str)
    args = parser.parse_args()
    pooledDecoding(args.label,args.region,args.cluster)

