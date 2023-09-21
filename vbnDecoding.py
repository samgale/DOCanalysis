# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 15:25:55 2023

@author: svc_ccg
"""

import os, time, warnings
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
import h5py
import sklearn
from sklearn.svm import LinearSVC
import facemap.process


baseDir = r"C:\Users\svc_ccg\Desktop\Analysis\vbn"

stimTable = pd.read_csv(os.path.join(baseDir,'master_stim_table.csv'))

videoTable = pd.read_excel(os.path.join(baseDir,'vbn_video_paths_full_validation.xlsx'))
videoTable.insert(0,'session_id',[int(s[:s.find('_')]) for s in videoTable['exp_id']])

sessionIds = stimTable['session_id'].unique()


dlcLabel = 'nose_tip'
roiParams = {'side': {'xoffset': -10, 'yoffset': -60, 'width': 110, 'height': 110},
             'face': {'xoffset': -90, 'yoffset': -160, 'width': 220, 'height': 220}}


# run facemap
dlcData = h5py.File(os.path.join(baseDir,'dlcData.hdf5'))
for sessionIndex,sessionId in enumerate(sessionIds):
    print(sessionIndex)
    for videoType,sbin in zip(('side','face'),(1,2)):    
        likelihood = dlcData[str(sessionId)][videoType][dlcLabel]['likelihood'][()]
        x,y = [int(np.average(dlcData[str(sessionId)][videoType][dlcLabel][c][()],weights=likelihood)) for c in ('x','y')]
        roi = np.array([x + roiParams[videoType]['xoffset'], y + roiParams[videoType]['yoffset'],
                        roiParams[videoType]['width'], roiParams[videoType]['height']])
        
        videoIndex = np.where(videoTable['session_id'] == sessionId)[0][0]
        videoPath = videoTable.loc[videoIndex,videoType+'_video']
        facemapSavePath = os.path.join(baseDir,'facemapOutput')
        
        proc = {'sx': np.array([0]),
                'sy': np.array([0]),
                'sbin': sbin,
                'fullSVD': False, # computes SVD for full video and roi if True else just roi
                'save_mat': False,
                'rois': [{'rind': 1,
                          'rtype': 'motion SVD',
                          'ivid': 0,
                          'xrange': np.arange(roi[0],roi[0]+roi[2]),
                          'yrange': np.arange(roi[1],roi[1]+roi[3])}],
                'savepath': facemapSavePath}
        
        facemap.process.run(filenames=[[videoPath]],
                            sbin=sbin,
                            motSVD=True,
                            movSVD=True, # throws an error if False
                            GUIobject=None,
                            parent=None,
                            proc=proc,
                            savepath=facemapSavePath)
dlcData.close()



f = r"C:\Users\svc_ccg\Desktop\Analysis\vbn\facemapOutput\1044385384_524761_20200819.face_proc.npy"

facemapData = np.load(f,allow_pickle=True).item()

d = {}
d['xrange'] = facemapData['rois'][0]['xrange']
d['yrange'] = facemapData['rois'][0]['yrange']
d['xrange_bin'] = facemapData['rois'][0]['xrange_bin']
d['yrange_bin'] = facemapData['rois'][0]['yrange_bin']
d['motSv'] = facemapData['motSv']
d['motSVD'] = facemapData['motSVD'][1]
d['motMask'] = facemapData['motMask_reshape'][1]

np.savez_compressed(f[:-3]+'npz',**d)


f = r"C:\Users\svc_ccg\Desktop\Analysis\vbn\facemapOutput\1044385384_524761_20200819.face_proc.npz"

d = np.load(f)


# decode licks from facemap SVDs
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


decodeData = {sessionId: {region: {layer: {} for layer in layers} for region in regions} for sessionId in sessionIds}
model = LinearSVC(C=1.0,max_iter=int(1e4))
nCrossVal = 5
decodeWindowEnd = 0.5
decodeWindows = np.arange(0,decodeWindowEnd,1/60)
warnings.filterwarnings('ignore')
for sessionIndex,sessionId in enumerate(sessionIds):
    print(sessionIndex)
    stim = stimTable[(stimTable['session_id']==sessionId) & stimTable['active']].reset_index()
    stimStart = stim['start_time'].iloc[0]
    stimEnd = stim['stop_time'].iloc[-1]
    flashTimes = np.array(stim['start_time'])
    changeTimes = flashTimes[stim['is_change']]
    hit = np.array(stim['hit'][stim['is_change']])
    engaged = np.array([np.sum(hit[(changeTimes>t-60) & (changeTimes<t+60)]) > 1 for t in flashTimes])
    autoRewarded = np.array(stim['auto_rewarded']).astype(bool)
    changeFlash = np.where(stim['is_change'] & ~autoRewarded & engaged)[0]
    nonChangeFlashes = np.array(engaged &
                                (~stim['is_change']) & 
                                (~stim['omitted']) & 
                                (~stim['previous_omitted']) & 
                                (stim['flashes_since_change']>5) &
                                (stim['flashes_since_last_lick']>1))
    lick = np.array(stim['lick_for_flash'])[nonChangeFlashes]
    nFlashes = nonChangeFlashes.sum()
    
    flashSvd = []
    videoIndex = np.where(videoTable['session_id'] == sessionId)[0][0]
    for videoType in ('side','face'):
        videoName = os.path.basename(videoTable.loc[videoIndex,videoType+'_video'])
        facemapDataPath = os.path.join(baseDir,'facemapOutput',videoName[:-4]+'_proc.npz')
        facemapData = np.load(facemapDataPath)
        svd = facemapData['motSVD']
        frameTimesPath = videoTable.loc[videoIndex,videoType+'_timestamp_path']
        frameTimes = np.load(frameTimesPath)
        flashSvd.append([])
        for flashTime in flashTimes[nonChangeFlashes]:
            frameIndex = np.searchsorted(frameTimes,decodeWindows+flashTime)
            flashSvd[-1].append(svd[frameIndex])
    flashSvd = np.concatenate(flashSvd,axis=2)
    
    trainAccuracy = []
    featureWeights = []
    accuracy = []
    balancedAccuracy = []
    prediction = []
    confidence = []
    for i in range(len(decodeWindows)):
        print(i)
        X = flashSvd[:,:i+1].reshape(nFlashes,-1)                       
        cv = crossValidate(model,X,lick,nCrossVal)
        trainAccuracy.append(np.mean(cv['train_score']))
        featureWeights.append(np.mean(cv['coef'],axis=0).squeeze())
        accuracy.append(np.mean(cv['test_score']))
        balancedAccuracy.append(sklearn.metrics.balanced_accuracy_score(lick,cv['predict']))
        prediction.append(cv['predict'])
        confidence.append(cv['decision_function'])
warnings.filterwarnings('default')



warnings.filterwarnings('ignore')
for sessionIndex,sessionId in enumerate(sessionIds):
    units = unitTable.set_index('unit_id').loc[unitData[str(sessionId)]['unitIds'][:]]
    spikes = unitData[str(sessionId)]['spikes']
    stim = stimTable[(stimTable['session_id']==sessionId) & stimTable['active']].reset_index()
    flashTimes = np.array(stim['start_time'])
    changeTimes = flashTimes[stim['is_change']]
    hit = np.array(stim['hit'][stim['is_change']])
    engaged = np.array([np.sum(hit[(changeTimes>t-60) & (changeTimes<t+60)]) > 1 for t in flashTimes])
    autoRewarded = np.array(stim['auto_rewarded']).astype(bool)
    changeFlash = np.where(stim['is_change'] & ~autoRewarded & engaged)[0]
    nonChangeFlashes = np.array(engaged &
                                (~stim['is_change']) & 
                                (~stim['omitted']) & 
                                (~stim['previous_omitted']) & 
                                (stim['flashes_since_change']>5) &
                                (stim['flashes_since_last_lick']>1))
    lick = np.array(stim['lick_for_flash'])[nonChangeFlashes]
    
    lickDecodeData[sessionId]['image'] = np.array(stim['image_name'])[nonChangeFlashes]
    lickDecodeData[sessionId]['flashesSinceLick'] = np.array(stim['flashes_since_last_lick'])[nonChangeFlashes]
    lickDecodeData[sessionId]['lick'] = lick
    
    nFlashes = nonChangeFlashes.sum()
    for region in regions:
        inRegion = np.in1d(units['structure_acronym'],region)
        if not any(inRegion):
            continue
        for layer in ('all',):
            print('session '+str(sessionIndex+1)+', '+str(region)+', '+str(layer))
            if layer=='all':
                inLayer = inRegion
            elif 'VIS' in region:
                inLayer = inRegion & np.in1d(units['cortical_layer'],layer)
            else:
                continue
            if not any(inLayer):
                continue
            sp = np.zeros((inLayer.sum(),spikes.shape[1],spikes.shape[2]),dtype=bool)
            for i,u in enumerate(np.where(inLayer)[0]):
                sp[i]=spikes[u,:,:]
                
            changeSp = sp[:,changeFlash,:]
            preChangeSp = sp[:,changeFlash-1,:]
            hasResp = findResponsiveUnits(preChangeSp,changeSp,baseWin,respWin)
            nUnits = hasResp.sum()
            if nUnits < sampleSize:
                continue
            flashSp = sp[hasResp][:,nonChangeFlashes,:decodeWindows[-1]].reshape((nUnits,nFlashes,len(decodeWindows),decodeWindowSize)).sum(axis=-1)
            
            if sampleSize>1:
                if sampleSize==nUnits:
                    nSamples = 1
                    unitSamples = [np.arange(nUnits)]
                else:
                    # >99% chance each neuron is chosen at least once
                    nSamples = int(math.ceil(math.log(0.01)/math.log(1-sampleSize/nUnits)))
                    unitSamples = [np.random.choice(nUnits,sampleSize,replace=False) for _ in range(nSamples)]
            else:
                nSamples = nUnits
                unitSamples = [[i] for i in range(nUnits)]
            trainAccuracy = np.full((len(unitSamples),len(decodeWindows)),np.nan)
            featureWeights = np.full((len(unitSamples),len(decodeWindows),nUnits,len(decodeWindows)),np.nan)
            accuracy = trainAccuracy.copy()
            balancedAccuracy = accuracy.copy()
            prediction = np.full((len(unitSamples),len(decodeWindows),nFlashes),np.nan)
            confidence = prediction.copy()
            for i,unitSamp in enumerate(unitSamples):
                for j,winEnd in enumerate((decodeWindows/decodeWindowSize).astype(int)):
                    X = flashSp[unitSamp,:,:winEnd].transpose(1,0,2).reshape((nFlashes,-1))                        
                    cv = crossValidate(model,X,lick,nCrossVal)
                    trainAccuracy[i,j] = np.mean(cv['train_score'])
                    featureWeights[i,j,unitSamp,:winEnd] = np.mean(cv['coef'],axis=0).reshape(sampleSize,winEnd)
                    accuracy[i,j] = np.mean(cv['test_score'])
                    balancedAccuracy[i,j] = sklearn.metrics.balanced_accuracy_score(lick,cv['predict'])
                    prediction[i,j] = cv['predict']
                    confidence[i,j] = cv['decision_function']
            lickDecodeData[sessionId][region][layer]['trainAccuracy'] = np.median(trainAccuracy,axis=0)
            lickDecodeData[sessionId][region][layer]['featureWeights'] = np.nanmedian(featureWeights,axis=0)
            lickDecodeData[sessionId][region][layer]['accuracy'] = np.median(accuracy,axis=0)
            lickDecodeData[sessionId][region][layer]['balancedAccuracy'] = np.median(balancedAccuracy,axis=0) 
            lickDecodeData[sessionId][region][layer]['prediction'] = scipy.stats.mode(prediction,axis=0)[0][0]
            lickDecodeData[sessionId][region][layer]['confidence'] = np.median(confidence,axis=0)
warnings.filterwarnings('default')






