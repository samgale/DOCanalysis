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
from vbnAnalysisUtils import findNearest, getBehavData, findResponsiveUnits, crossValidate


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
    flashTimes,changeFlashes,nonChangeFlashes,lick,lickTimes,hit = getBehavData(stim)
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
        cv = crossValidate(model,X,y,nCrossVal)
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
    decodeWindowSize = 10
    decodeWindowEnd = 750
    decodeWindows = np.arange(decodeWindowSize,decodeWindowEnd+decodeWindowSize,decodeWindowSize)

    regions = ('LGd','VISp','VISl','VISrl','VISal','VISpm','VISam','LP',
               'MRN','MB',('SCig','SCiw'),'APN','NOT',
               ('HPF','DG','CA1','CA3'),('SUB','ProS','PRE','POST'))

    baseWin = slice(680,750)
    respWin = slice(30,100)

    units = unitTable.set_index('unit_id').loc[unitData[str(sessionId)]['unitIds'][:]]
    spikes = unitData[str(sessionId)]['spikes']
    
    stim = stimTable[(stimTable['session_id']==sessionId) & stimTable['active']].reset_index()
    flashTimes,changeFlashes,nonChangeFlashes,lick,lickTimes,hit = getBehavData(stim)
    
    d = {region: {sampleSize: {metric: [] for metric in ('trainAccuracy','featureWeights','accuracy','balancedAccuracy','prediction','confidence')}
         for sampleSize in unitSampleSize} for region in regions}
    d['decodeWindows'] = decodeWindows
    d['lick'] = lick[nonChangeFlashes]
    y = lick[nonChangeFlashes]
    warnings.filterwarnings('ignore')
    for region in regions:
        inRegion = np.in1d(units['structure_acronym'],region)
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
                if sampleSize!=20 and winEnd!=decodeWindows[-1]:
                    continue
                winEnd = int(winEnd/decodeWindowSize)
                for metric in d[region][sampleSize]:
                    d[region][sampleSize][metric].append([])
                for unitSamp in unitSamples:
                    X = flashSp[unitSamp,:,:winEnd].transpose(1,0,2).reshape((nonChangeFlashes.sum(),-1))                        
                    cv = crossValidate(model,X,y,nCrossVal)
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
               'MRN','MB',('SCig','SCiw'),'APN','NOT',
               ('HPF','DG','CA1','CA3'),('SUB','ProS','PRE','POST'))

    baseWin = slice(680,750)
    respWin = slice(30,100)

    units = unitTable.set_index('unit_id').loc[unitData[str(sessionId)]['unitIds'][:]]
    spikes = unitData[str(sessionId)]['spikes']
    
    stim = stimTable[(stimTable['session_id']==sessionId) & stimTable['active']].reset_index()
    flashTimes,changeFlashes,nonChangeFlashes,lick,lickTimes,hit = getBehavData(stim)
    nChange = changeFlashes.sum()
    
    d = {region: {sampleSize: {metric: [] for metric in ('trainAccuracy','featureWeights','accuracy','prediction','confidence')}
         for sampleSize in unitSampleSize} for region in regions}
    d['decodeWindows'] = decodeWindows
    d['hit'] = hit[changeFlashes]
    y = np.zeros(nChange*2)
    y[:nChange] = 1
    warnings.filterwarnings('ignore')
    for region in regions:
        inRegion = np.in1d(units['structure_acronym'],region)
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
                if sampleSize!=20 and winEnd!=decodeWindows[-1]:
                    continue
                winEnd = int(winEnd/decodeWindowSize)
                for metric in d[region][sampleSize]:
                    d[region][sampleSize][metric].append([])
                for unitSamp in unitSamples:
                    X = np.concatenate([s[unitSamp,:,:winEnd].transpose(1,0,2).reshape((nChange,-1)) for s in (changeSp,preChangeSp)])                       
                    cv = crossValidate(model,X,y,nCrossVal)
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
    

def predictResposeTimesFromDecoder(sessionId):
    model = LinearSVC(C=1.0,max_iter=int(1e4),class_weight=None)
    minUnits = 20
    decodeWindowSize = 10
    decodeWindowEnd = 100
    decodeWindows = np.arange(decodeWindowSize,decodeWindowEnd+decodeWindowSize,decodeWindowSize)

    regions = ('VISp','VISam')

    baseWin = slice(680,750)
    respWin = slice(30,100)

    units = unitTable.set_index('unit_id').loc[unitData[str(sessionId)]['unitIds'][:]]
    spikes = unitData[str(sessionId)]['spikes']
    
    stim = stimTable[(stimTable['session_id']==sessionId) & stimTable['active']].reset_index()
    flashTimes,changeFlashes,nonChangeFlashes,lick,lickTimes,hit = getBehavData(stim)
    nChange = changeFlashes.sum()
    
    d = {region: {metric: [] for metric in ('trainAccuracy','featureWeights','prediction','confidence')} for region in regions}
    d['sessionId'] = sessionId
    d['regions'] = regions
    d['decodeWindows'] = decodeWindows
    d['hit'] = hit[changeFlashes]
    d['preChangeImage'] = np.array(stim['initial_image_name'])[changeFlashes]
    d['changeImage'] = np.array(stim['change_image_name'])[changeFlashes]
    d['novelImage'] = np.array(stim['novel_image']).astype(bool)
    y = np.zeros(nChange*2)
    y[:nChange] = 1
    warnings.filterwarnings('ignore')
    for region in regions:
        inRegion = np.in1d(units['structure_acronym'],region)
        if not any(inRegion):
            continue
                
        sp = np.zeros((inRegion.sum(),spikes.shape[1],spikes.shape[2]),dtype=bool)
        for i,u in enumerate(np.where(inRegion)[0]):
            sp[i]=spikes[u,:,:]
            
        changeSp = sp[:,changeFlashes,:]
        preChangeSp = sp[:,np.where(changeFlashes)[0]-1,:]
        hasResp = findResponsiveUnits(preChangeSp,changeSp,baseWin,respWin)
        nUnits = hasResp.sum()
        if nUnits < minUnits:
            continue
        
        X = np.concatenate([s[:,:,:decodeWindowEnd].sum(axis=-1).T for s in (changeSp,preChangeSp)])
        d[region]['prediction'] = np.zeros(nChange*2)
        for trial in range(nChange):
            estimator = sklearn.base.clone(model)
            trainInd = [tr for tr in range(nChange*2) if tr not in (trial,trial+nChange)]
            estimator.fit(X[trainInd],y[trainInd])
            d[region]['trainAccuracy'].append(estimator.score(X[trainInd],y[trainInd]))
            d[region]['featureWeights'].append(estimator.coef_.squeeze())
            d[region]['prediction'][[trial,trial+nChange]] = estimator.predict(X[[trial,trial+nChange]])
            d[region]['confidence'].append([])
            for winEnd in decodeWindows:
                trialX = changeSp[:,trial,:winEnd].sum(axis=-1).reshape(1,-1)
                d[region]['confidence'][-1].append(estimator.decision_function(trialX)[0])
        d[region]['accuracy'] = sklearn.metrics.accuracy_score(y,d[region]['prediction'])
    warnings.filterwarnings('default')

    np.save(os.path.join(outputDir,'decoderResponseTimes','decoderResponseTimes_'+str(sessionId)+'.npy'),d)

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--sessionId',type=int)
    args = parser.parse_args()
    #runFacemap(args.sessionId)
    #decodeLicksFromFacemap(args.sessionId)
    #decodeLicksFromUnits(args.sessionId)
    #decodeChange(args.sessionId)
    predictResposeTimesFromDecoder(args.sessionId)
