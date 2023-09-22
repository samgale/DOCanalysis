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

roiParams = {'side': {'xoffset': -5, 'yoffset': -10, 'width': 40, 'height': 40},
             'face': {'xoffset': -35, 'yoffset': -50, 'width': 80, 'height': 80}}


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
        
        # video = cv2.VideoCapture(videoPath)
        # video.read()
        # image = video.read()[1]
        # plt.figure()
        # plt.imshow(image[roi[1]:roi[1]+roi[3],roi[0]:roi[0]+roi[2],0],cmap='gray')
        # video.release()
        
        facemapSavePath = os.path.join(baseDir,'facemapOutput')
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
        
        h5Out = h5py.File(os.path.join(baseDir,'facemapOutput',facemapDataPath[:-3]+'hdf5'),'w')
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
        for key,val in d.items():
            h5Out.create_dataset(key,data=val,compression='gzip',compression_opts=4)
        h5Out.close()
        # np.savez_compressed(facemapDataPath[:-3]+'npz',**d)
        # os.remove(facemapDataPath)
dlcData.close()


#
fig = plt.figure()
gs = matplotlib.gridspec.GridSpec(2,3)
for i,videoType in enumerate(('side','face')):
    videoIndex = np.where(videoTable['session_id'] == sessionId)[0][0]
    videoName = os.path.basename(videoTable.loc[videoIndex,videoType+'_video'])
    facemapDataPath = os.path.join(baseDir,'facemapOutput',videoName[:-4]+'_proc.hdf5')
    facemapData = h5py.File(facemapDataPath)
    x = facemapData['xrange_bin'][()]
    y = facemapData['yrange_bin'][()]
    
    ax = fig.add_subplot(gs[i,0])
    ax.imshow(facemapData['avgframe'][()][y,:][:,x],cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    if i==0:
        ax.set_title('average frame')
    
    ax = fig.add_subplot(gs[i,1])
    ax.imshow(facemapData['avgmotion'][()][y,:][:,x],cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    if i==0:
        ax.set_title('average motion')
    
    ax = fig.add_subplot(gs[i,2])
    ax.imshow(np.average(facemapData['motMask'][()],weights=facemapData['motSv'][()],axis=2),cmap='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    if i==0:
        ax.set_title('weighted average\nSVD projection')
    
    facemapData.close()
plt.tight_layout()


#
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
for sessionId in sessionIds: [sessionIds[0]]
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
    lick = np.array(stim['lick_for_flash'])
    lickTimes = np.array(stim['lick_time'])
    lickLatency = lickTimes - flashTimes
    earlyLick = lickLatency < 0.15
    lateLick = lickLatency > 0.75
    d = lickLatency[nonChangeFlashes & lick & ~earlyLick & ~lateLick]
    dsort = np.sort(lickLatency)
    cumProb = np.array([np.sum(d<=i)/d.size for i in dsort])
    ax.plot(dsort,cumProb,'k',alpha=0.25)
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_xlim([0,0.75])
ax.set_ylim([0,1.01])
ax.set_xlabel('Lick latency (s)')
ax.set_ylabel('Cumulative probability')
plt.tight_layout()


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
decodeWindowEnd = 0.75
frameInterval = 1/60
decodeWindows = np.arange(0,decodeWindowEnd+frameInterval/2,frameInterval)
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
    lick = np.array(stim['lick_for_flash'])
    lickTimes = np.array(stim['lick_time'])
    lickLatency = lickTimes - flashTimes
    earlyLick = lickLatency < 0.15
    nFlashes = np.sum(nonChangeFlashes & ~earlyLick)
    
    flashSvd = []
    videoIndex = np.where(videoTable['session_id'] == sessionId)[0][0]
    for videoType in ('side','face'):
        videoName = os.path.basename(videoTable.loc[videoIndex,videoType+'_video'])
        facemapDataPath = os.path.join(baseDir,'facemapOutput',videoName[:-4]+'_proc.hdf5')
        with h5py.File(facemapDataPath) as facemapData:
            svd = facemapData['motSVD'][()]
        frameTimesPath = videoTable.loc[videoIndex,videoType+'_timestamp_path']
        frameTimes = np.load(frameTimesPath)
        flashSvd.append([])
        for flashTime in flashTimes[nonChangeFlashes & ~earlyLick]:
            frameIndex = np.searchsorted(frameTimes,decodeWindows+flashTime)
            flashSvd[-1].append(svd[frameIndex])
    flashSvd = np.concatenate(flashSvd,axis=2)
    
    trainAccuracy = []
    featureWeights = []
    accuracy = []
    balancedAccuracy = []
    prediction = []
    confidence = []
    y = lick[nonChangeFlashes & ~earlyLick] 
    for i in range(len(decodeWindows)):
        print(i)
        X = flashSvd[:,:i+1].reshape(nFlashes,-1)   
        cv = crossValidate(model,X,y,nCrossVal)
        trainAccuracy.append(np.mean(cv['train_score']))
        featureWeights.append(np.mean(cv['coef'],axis=0).squeeze())
        accuracy.append(np.mean(cv['test_score']))
        balancedAccuracy.append(sklearn.metrics.balanced_accuracy_score(y,cv['predict']))
        prediction.append(cv['predict'])
        confidence.append(cv['decision_function'])
warnings.filterwarnings('default')


fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(decodeWindows,balancedAccuracy,'k')
for side in ('right','top'):
    ax.spines[side].set_visible(False)
ax.tick_params(direction='out',top=False,right=False)
ax.set_ylim([0.5,1])
ax.set_xlabel('Time from non-change flash onset (ms)')
ax.set_ylabel('Lick decoding balanced accuracy')
plt.tight_layout()




# decode licks from dlc data
decodeData = {sessionId: {region: {layer: {} for layer in layers} for region in regions} for sessionId in sessionIds}
model = LinearSVC(C=1.0,max_iter=int(1e4))
nCrossVal = 5
decodeWindowEnd = 0.75
frameInterval = 1/60
decodeWindows = np.arange(0,decodeWindowEnd+frameInterval/2,frameInterval)
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
    lick = np.array(stim['lick_for_flash'])
    lickTimes = np.array(stim['lick_time'])
    lickLatency = lickTimes - flashTimes
    earlyLick = lickLatency < 0.15
    nFlashes = np.sum(nonChangeFlashes & ~earlyLick)
    
    features = []
    videoIndex = np.where(videoTable['session_id'] == sessionId)[0][0]
    for videoType in ('side','face'):
        frameTimesPath = videoTable.loc[videoIndex,videoType+'_timestamp_path']
        frameTimes = np.load(frameTimesPath)
        featureNames = list(dlcData[str(sessionId)][videoType].keys())
        a = []
        for key in featureNames:
            a.append([])
            for c in ('x','y'):
                a[-1].append([])
                d = dlcData[str(sessionId)][videoType][key][c][()]
                for flashTime in flashTimes[nonChangeFlashes & ~earlyLick]:
                    frameIndex = np.searchsorted(frameTimes,decodeWindows+flashTime)
                    a[-1][-1].append(d[frameIndex])
        features.append(np.array(a).reshape((len(featureNames)*2,nFlashes,len(decodeWindows))).transpose((1,2,0)))
    features = np.concatenate(features,axis=-1)
    
    trainAccuracy = []
    featureWeights = []
    accuracy = []
    balancedAccuracy = []
    prediction = []
    confidence = []
    y = lick[nonChangeFlashes & ~earlyLick] 
    for i in range(len(decodeWindows)):
        print(i)
        X = features[:,:i+1].reshape(nFlashes,-1)   
        cv = crossValidate(model,X,y,nCrossVal)
        trainAccuracy.append(np.mean(cv['train_score']))
        featureWeights.append(np.mean(cv['coef'],axis=0).squeeze())
        accuracy.append(np.mean(cv['test_score']))
        balancedAccuracy.append(sklearn.metrics.balanced_accuracy_score(y,cv['predict']))
        prediction.append(cv['predict'])
        confidence.append(cv['decision_function'])
warnings.filterwarnings('default')













