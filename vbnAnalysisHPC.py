# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 14:37:16 2022

@author: svc_ccg
"""

import argparse
import math
import os
import warnings
import numpy as np
import pandas as pd
import scipy.stats
import h5py
import sklearn
from sklearn.svm import LinearSVC
import facemap.process
from vbnAnalysisUtils import dictToHdf5, findNearest, getBehavData, getUnitsInRegion, findResponsiveUnits, trainDecoder


baseDir = '/allen/programs/mindscope/workgroups/np-behavior/vbn_data_release/supplemental_tables'

outputDir = '/allen/programs/mindscope/workgroups/np-behavior/VBN_video_analysis'

stimTable = pd.read_csv(os.path.join(baseDir,'master_stim_table.csv'))

videoTable = pd.read_excel(os.path.join(baseDir,'vbn_video_paths_full_validation.xlsx'))
videoTable.insert(0,'session_id',[int(s[:s.find('_')]) for s in videoTable['exp_id']])

unitTable = pd.read_csv(os.path.join(baseDir,'units_with_cortical_layers.csv'))
unitData = h5py.File(os.path.join(baseDir,'vbnAllUnitSpikeTensor.hdf5'),mode='r')


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


def decodeLicksFromFacemap(sessionId):
    model = LinearSVC(C=1.0,max_iter=int(1e4),class_weight=None)
    nCrossVal = 5
    decodeWindowStart = 0
    decodeWindowEnd = 0.75
    frameInterval = 1/60
    decodeWindows = np.arange(decodeWindowStart,decodeWindowEnd+frameInterval/2,frameInterval)
    
    stim = stimTable[(stimTable['session_id']==sessionId) & stimTable['active']].reset_index()
    flashTimes,changeFlashes,catchFlashes,nonChangeFlashes,omittedFlashes,prevOmittedFlashes,novelFlashes,lick,lickTimes = getBehavData(stim)
    flashTimes = flashTimes[nonChangeFlashes]
    lick = lick[nonChangeFlashes]
    
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
    
    d = {metric: [] for metric in ('trainAccuracy','featureWeights','accuracy','balancedAccuracy','prediction','confidence')}
    d['decodeWindows'] = decodeWindows
    d['lick'] = lick
    y = lick
    warnings.filterwarnings('ignore')
    for i in range(len(decodeWindows)):
        X = flashSvd[:,:i+1].reshape(len(flashTimes),-1)   
        cv = trainDecoder(model,X,y,nCrossVal)
        d['trainAccuracy'].append(np.mean(cv['train_score']))
        d['featureWeights'].append(np.mean(cv['coef'],axis=0).squeeze())
        d['accuracy'].append(np.mean(cv['test_score']))
        d['balancedAccuracy'].append(sklearn.metrics.balanced_accuracy_score(y,cv['predict']))
        d['prediction'].append(cv['predict'])
        d['confidence'].append(cv['decision_function'])
    warnings.filterwarnings('default')
    
    np.save(os.path.join(outputDir,'facemapLickDecoding','facemapLickDecoding_'+str(sessionId)+'.npy'),d)


def decodeLicksFromUnits(sessionId):
    model = LinearSVC(C=1.0,max_iter=int(1e4),class_weight=None)
    nCrossVal = 5
    unitSampleSize = [1,5,10,15,20,25,30,40,50,60]
    decodeWindowSampleSize = 10
    decodeWindowSize = 10
    decodeWindowEnd = 750
    decodeWindows = np.arange(decodeWindowSize,decodeWindowEnd+decodeWindowSize,decodeWindowSize)

    regions = ('LGd','VISp','VISl','VISrl','VISal','VISpm','VISam','LP',
               'MRN','MB','SC','APN','NOT','Hipp','Sub',
               'SC/MRN cluster 1','SC/MRN cluster 2')

    baseWin = slice(680,750)
    respWin = slice(30,100)

    units = unitTable.set_index('unit_id').loc[unitData[str(sessionId)]['unitIds'][:]]
    spikes = unitData[str(sessionId)]['spikes']
    
    stim = stimTable[(stimTable['session_id']==sessionId) & stimTable['active']].reset_index()
    flashTimes,changeFlashes,catchFlashes,nonChangeFlashes,omittedFlashes,prevOmittedFlashes,novelFlashes,lick,lickTimes = getBehavData(stim)
    
    d = {region: {sampleSize: {metric: [] for metric in ('trainAccuracy','featureWeights','accuracy','balancedAccuracy','prediction','confidence')}
         for sampleSize in unitSampleSize} for region in regions}
    d['decodeWindows'] = decodeWindows
    d['lick'] = lick[nonChangeFlashes]
    y = lick[nonChangeFlashes]
    warnings.filterwarnings('ignore')
    for region in regions:
        inRegion = getUnitsInRegion(units,region)
        if not any(inRegion):
            continue
                
        sp = np.zeros((inRegion.sum(),spikes.shape[1],spikes.shape[2]),dtype=bool)
        for i,u in enumerate(np.where(inRegion)[0]):
            sp[i]=spikes[u,:,:]
            
        changeSp = sp[:,changeFlashes,:]
        preChangeSp = sp[:,np.where(changeFlashes)[0]-1,:]
        hasResp = findResponsiveUnits(preChangeSp,changeSp,baseWin,respWin)
        nUnits = hasResp.sum()

        flashSp = sp[hasResp][:,nonChangeFlashes,:decodeWindows[-1]].reshape((nUnits,nonChangeFlashes.sum(),len(decodeWindows),decodeWindowSize)).sum(axis=-1)
        
        for sampleSize in unitSampleSize:
            if nUnits < sampleSize:
                continue
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

            for winEnd in decodeWindows:
                if sampleSize!=decodeWindowSampleSize and winEnd!=decodeWindows[-1]:
                    continue
                winEnd = int(winEnd/decodeWindowSize)
                for metric in d[region][sampleSize]:
                    d[region][sampleSize][metric].append([])
                for unitSamp in unitSamples:
                    X = flashSp[unitSamp,:,:winEnd].transpose(1,0,2).reshape((nonChangeFlashes.sum(),-1))                        
                    cv = trainDecoder(model,X,y,nCrossVal)
                    d[region][sampleSize]['trainAccuracy'][-1].append(np.mean(cv['train_score']))
                    d[region][sampleSize]['featureWeights'][-1].append(np.mean(cv['coef'],axis=0).squeeze())
                    d[region][sampleSize]['accuracy'][-1].append(np.mean(cv['test_score']))
                    d[region][sampleSize]['balancedAccuracy'][-1].append(sklearn.metrics.balanced_accuracy_score(y,cv['predict']))
                    d[region][sampleSize]['prediction'][-1].append(cv['predict'])
                    d[region][sampleSize]['confidence'][-1].append(cv['decision_function'])
                for metric in d[region][sampleSize]:
                    if metric == 'prediction':
                        d[region][sampleSize][metric][-1] = scipy.stats.mode(d[region][sampleSize][metric][-1],axis=0)[0][0]
                    else:
                        d[region][sampleSize][metric][-1] = np.median(d[region][sampleSize][metric][-1],axis=0)
    warnings.filterwarnings('default')

    np.save(os.path.join(outputDir,'unitLickDecoding','unitLickDecoding_'+str(sessionId)+'.npy'),d)


def decodeChange(sessionId):
    model = LinearSVC(C=1.0,max_iter=int(1e4),class_weight=None)
    nCrossVal = 5
    unitSampleSize = [1,5,10,15,20,25,30,40,50,60]
    decodeWindowSize = 10
    decodeWindowEnd = 750
    decodeWindows = np.arange(decodeWindowSize,decodeWindowEnd+decodeWindowSize,decodeWindowSize)

    regions = ('LGd','VISp','VISl','VISrl','VISal','VISpm','VISam','LP',
               'MRN','MB','SC','APN','NOT','Hipp','Sub',
               'SC/MRN cluster 1','SC/MRN cluster 2')

    baseWin = slice(680,750)
    respWin = slice(30,100)

    units = unitTable.set_index('unit_id').loc[unitData[str(sessionId)]['unitIds'][:]]
    spikes = unitData[str(sessionId)]['spikes']
    
    stim = stimTable[(stimTable['session_id']==sessionId) & stimTable['active']].reset_index()
    flashTimes,changeFlashes,catchFlashes,nonChangeFlashes,omittedFlashes,prevOmittedFlashes,novelFlashes,lick,lickTimes = getBehavData(stim)
    nChange = changeFlashes.sum()
    
    d = {region: {sampleSize: {metric: [] for metric in ('trainAccuracy','featureWeights','accuracy','prediction','confidence')}
         for sampleSize in unitSampleSize} for region in regions}
    d['decodeWindows'] = decodeWindows
    d['hit'] = np.array(stim['hit'])[changeFlashes]
    y = np.zeros(nChange*2)
    y[:nChange] = 1
    warnings.filterwarnings('ignore')
    for region in regions:
        inRegion = getUnitsInRegion(units,region)
        if not any(inRegion):
            continue
                
        sp = np.zeros((inRegion.sum(),spikes.shape[1],spikes.shape[2]),dtype=bool)
        for i,u in enumerate(np.where(inRegion)[0]):
            sp[i]=spikes[u,:,:]
            
        changeSp = sp[:,changeFlashes,:]
        preChangeSp = sp[:,np.where(changeFlashes)[0]-1,:]
        hasResp = findResponsiveUnits(preChangeSp,changeSp,baseWin,respWin)
        nUnits = hasResp.sum()

        changeSp,preChangeSp = [s[hasResp,:,:decodeWindows[-1]].reshape((nUnits,nChange,len(decodeWindows),decodeWindowSize)).sum(axis=-1) for s in (changeSp,preChangeSp)]
        
        decodeWindowSampleSize = 10 if region=='SC/MRN cluster 1' else 20
        for sampleSize in unitSampleSize:
            if nUnits < sampleSize:
                continue
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

            for winEnd in decodeWindows:
                if sampleSize!=decodeWindowSampleSize and winEnd!=decodeWindows[-1]:
                    continue
                winEnd = int(winEnd/decodeWindowSize)
                for metric in d[region][sampleSize]:
                    d[region][sampleSize][metric].append([])
                for unitSamp in unitSamples:
                    X = np.concatenate([s[unitSamp,:,:winEnd].transpose(1,0,2).reshape((nChange,-1)) for s in (changeSp,preChangeSp)])                       
                    cv = trainDecoder(model,X,y,nCrossVal)
                    d[region][sampleSize]['trainAccuracy'][-1].append(np.mean(cv['train_score']))
                    d[region][sampleSize]['featureWeights'][-1].append(np.mean(cv['coef'],axis=0).squeeze())
                    d[region][sampleSize]['accuracy'][-1].append(np.mean(cv['test_score']))
                    d[region][sampleSize]['prediction'][-1].append(cv['predict'])
                    d[region][sampleSize]['confidence'][-1].append(cv['decision_function'])
                for metric in d[region][sampleSize]:
                    if metric == 'prediction':
                        d[region][sampleSize][metric][-1] = scipy.stats.mode(d[region][sampleSize][metric][-1],axis=0)[0][0]
                    else:
                        d[region][sampleSize][metric][-1] = np.median(d[region][sampleSize][metric][-1],axis=0)
    warnings.filterwarnings('default')

    np.save(os.path.join(outputDir,'unitChangeDecoding','unitChangeDecoding_'+str(sessionId)+'.npy'),d)


def fitIntegratorModel(sessionId):
    regions = ('LGd','VISp','VISl','VISrl','VISal','VISpm','VISam','VISall','SC/MRN cluster 2')
    nCrossVal = 5
    minUnits = 20
    baseWin = slice(680,750)
    respWin = slice(30,100)
    tEnd = 150
    leakRange= np.arange(0,1.01,0.01)
    thresholdRange = np.arange(0.1,10.1,0.1)

    units = unitTable.set_index('unit_id').loc[unitData[str(sessionId)]['unitIds'][:]]
    rsUnits = np.array(units['waveform_duration'] > 0.4)
    spikes = unitData[str(sessionId)]['spikes']
    
    stim = stimTable[(stimTable['session_id']==sessionId) & stimTable['active']].reset_index()
    flashTimes,changeFlashes,catchFlashes,nonChangeFlashes,omittedFlashes,prevOmittedFlashes,novelFlashes,lick,lickTimes = getBehavData(stim)
    preChangeFlashes = np.concatenate((changeFlashes[1:],[False]))
    nFlash = len(flashTimes)
    nChange = changeFlashes.sum()
    
    d = {region: {} for region in regions}
    d['sessionId'] = sessionId
    d['regions'] = regions
    d['leakRange'] = leakRange
    d['thresholdRange'] = thresholdRange
    d['imageName'] = np.array(stim['image_name'])
    d['change'] = changeFlashes
    d['preChange'] = preChangeFlashes
    d['catch'] = catchFlashes
    d['nonChange'] = nonChangeFlashes
    d['omitted'] = omittedFlashes
    d['prevOmitted'] = prevOmittedFlashes
    d['novel'] = novelFlashes
    d['lick'] = lick
    d['lickLatency'] = lickTimes - flashTimes
    outcome = []
    for lbl in ('hit','miss','false_alarm','correct_reject'):
        a = stim[lbl].copy()
        a[a.isnull()] = False
        outcome.append(np.array(a).astype(bool))
    d['hit'] = outcome[0] & changeFlashes
    d['miss'] = outcome[1] & changeFlashes
    d['falseAlarm'] = outcome[2] & catchFlashes
    d['correctReject'] = outcome[3] & catchFlashes

    warnings.filterwarnings('ignore')
    for region in regions:
        inRegion = getUnitsInRegion(units,region,rs=True)
        if not any(inRegion):
            continue
        
        sp = np.zeros((inRegion.sum(),spikes.shape[1],spikes.shape[2]),dtype=bool)
        for i,u in enumerate(np.where(inRegion)[0]):
            sp[i]=spikes[u,:,:]
            
        changeSp = sp[:,changeFlashes,:]
        preChangeSp = sp[:,preChangeFlashes,:]
        hasResp = findResponsiveUnits(preChangeSp,changeSp,baseWin,respWin)
        nUnits = hasResp.sum()
        if nUnits < minUnits:
            continue
        d[region]['nUnits'] = nUnits
        
        flashSp = sp[hasResp].mean(axis=0)
        baseSpRate = flashSp[:,baseWin].mean(axis=1)
        baseSpRate[1:] = baseSpRate[:-1]
        flashSp -= baseSpRate[:,None]
        flashSp = flashSp[:,1:tEnd]
        maxSpRate = flashSp[changeFlashes | preChangeFlashes].max()
        d[region]['spikeRate'] = flashSp*1000
        d[region]['baseSpikeRate'] = baseSpRate[changeFlashes].mean()*1000
        d[region]['maxSpikeRate'] = maxSpRate*1000
        flashSp /= maxSpRate
        
        y = np.full(nFlash,np.nan)
        y[changeFlashes] = 1
        y[preChangeFlashes] = 0
        samplesPerClassPerSplit = round(nChange/nCrossVal)
        shuffleInd = np.random.permutation(np.where(changeFlashes | preChangeFlashes)[0])
        leakFit = np.zeros(nCrossVal+1)
        thresholdFit = np.zeros(nCrossVal+1)
        trainAccuracy = np.zeros((nCrossVal+1,leakRange.size,thresholdRange.size))
        integrator = np.zeros((nFlash,tEnd))
        modelResp = np.zeros(nFlash)
        modelRespTime = np.full(nFlash,np.nan)
        trainInd = []
        testInd = []
        for k in range(nCrossVal):
            testInd.append([])
            for val in (0,1):
                i = k*samplesPerClassPerSplit
                ind = shuffleInd[y[shuffleInd]==val]
                testInd[-1].extend(ind[i:i+samplesPerClassPerSplit] if k<nCrossVal-1 else ind[i:])
            trainInd.append(np.setdiff1d(shuffleInd,testInd[-1]))
        trainInd.append(shuffleInd)
        testInd.append(np.where(~changeFlashes & ~preChangeFlashes)[0])
        for k,(train,test) in enumerate(zip(trainInd,testInd)):
            for i,leak in enumerate(leakRange):
                for j,thresh in enumerate(thresholdRange):
                    prediction = np.zeros(len(train))
                    for trial,trialSp in enumerate(flashSp[train]):
                        v = 0
                        for s in trialSp:
                            v += s - leak*v
                            if v > thresh:
                                prediction[trial] = 1
                                break
                    trainAccuracy[k,i,j] = sklearn.metrics.accuracy_score(y[train],prediction)
            i,j = np.unravel_index(np.argmax(trainAccuracy[k]),trainAccuracy[k].shape)
            leakFit[k] = leakRange[i]
            thresholdFit[k] = thresholdRange[j]
            for trial,trialSp in zip(test,flashSp[test]):
                for t,s in enumerate(trialSp):
                    integrator[trial,t+1] = integrator[trial,t] + s - leakFit[k]*integrator[trial,t]
                    if integrator[trial,t+1] > thresholdFit[k]:
                        modelResp[trial] = 1
                        modelRespTime[trial] = t+1
        d[region]['leak'] = leakFit
        d[region]['threshold'] = thresholdFit
        d[region]['trainAccuracy'] = trainAccuracy
        d[region]['integrator'] = integrator
        d[region]['modelResp'] = modelResp
        d[region]['modelRespTime'] = modelRespTime
        d[region]['modelAccuracy'] = sklearn.metrics.accuracy_score(y[shuffleInd],modelResp[shuffleInd])
        
        d[region]['decoderTrainAccuracy'] = {}
        d[region]['decoderWeights'] = {}
        d[region]['decoderAccuracy'] = {}
        d[region]['decoderPrediction'] = {}
        d[region]['decoderConfidence'] = {}
        for X,lbl in zip((flashSp,sp[hasResp,:,:tEnd].transpose(1,0,2).reshape((nFlash,-1))),('populationAverage','allUnits')):              
            decoderTrainAccuracy = np.zeros(nCrossVal+1)
            decoderWeights = np.zeros((nCrossVal+1,X.shape[1]))
            decoderAccuracy = np.zeros(nCrossVal)
            decoderPrediction = np.zeros(nFlash)
            decoderConfidence = np.zeros(nFlash)
            for k,(train,test) in enumerate(zip(trainInd,testInd)):
                decoder = LinearSVC(C=1.0,max_iter=int(1e4),class_weight=None)
                decoder.fit(X[train],y[train])
                decoderTrainAccuracy[k] = decoder.score(X[train],y[train])
                decoderWeights[k] = np.squeeze(decoder.coef_)
                if k < nCrossVal:
                    decoderAccuracy[k] = decoder.score(X[test],y[test])
                decoderPrediction[test] = decoder.predict(X[test])
                decoderConfidence[test] = decoder.decision_function(X[test])
            d[region]['decoderTrainAccuracy'][lbl] = decoderTrainAccuracy
            d[region]['decoderWeights'][lbl] = decoderWeights
            d[region]['decoderAccuracy'][lbl] = decoderAccuracy
            d[region]['decoderPrediction'][lbl] = decoderPrediction
            d[region]['decoderConfidence'][lbl] = decoderConfidence
    warnings.filterwarnings('default')

    h5File = h5py.File(os.path.join(outputDir,'integratorModel','integratorModel_'+str(sessionId)+'.hdf5'),'w')
    dictToHdf5(h5File,d)
    h5File.close()

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sessionId',type=int)
    args = parser.parse_args()
    sessionId = args.sessionId
    #runFacemap(sessionId)
    #decodeLicksFromFacemap(sessionId)
    decodeLicksFromUnits(sessionId)
    decodeChange(sessionId)
    #fitIntegratorModel(sessionId)
