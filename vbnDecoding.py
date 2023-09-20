# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 15:25:55 2023

@author: svc_ccg
"""

import os, time
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
import h5py
import skvideo
skvideo.setFFmpegPath(r"C:\Users\svc_ccg\Desktop\ffmpeg\bin") # run this before importing skvideo.io
import facemap.process


baseDir = r"C:\Users\svc_ccg\Desktop\Analysis\vbn"

stimTable = pd.read_csv(os.path.join(baseDir,'master_stim_table.csv'))

videoTable = pd.read_excel(os.path.join(baseDir,'vbn_video_paths_full_validation.xlsx'))
videoTable.insert(0,'session_id',[int(s[:s.find('_')]) for s in videoTable['exp_id']])

sessionIds = stimTable['session_id'].unique()


dlcLabel = 'nose_tip'
roiParams = {'side': {'xoffset': -10, 'yoffset': -60, 'width': 110, 'height': 110},
             'face': {'xoffset': -90, 'yoffset': -160, 'width': 220, 'height': 220}}


dlcData = h5py.File(os.path.join(baseDir,'dlcData.hdf5'))
alignedVideoFrameTimes = []
for sessionIndex,sessionId in enumerate(sessionIds):
    print(sessionIndex)
    stim = stimTable[(stimTable['session_id']==sessionId) & stimTable['active']].reset_index()
    stimStart = stim['start_time'].iloc[0]
    stimEnd = stim['stop_time'].iloc[-1]
    
    videoSessionIndex = []
    frameTimes = []
    frameIndex = []
    for videoType in ('side','face'):
        i = np.where(videoTable['session_id'] == sessionId)[0][0]
        videoSessionIndex.append(i)
        frameTimesPath = videoTable.loc[i,videoType+'_timestamp_path']
        ft = np.load(frameTimesPath)
        fi = np.where((ft >= stimStart) & (ft <= stimEnd))[0]
        frameTimes.append(ft[fi])
        frameIndex.append(fi)
    i,j = (0,1) if frameIndex[0].size >= frameIndex[1].size else (1,0)
    frameIndex[j] = frameIndex[j][np.searchsorted(frameTimes[j],frameTimes[i])[:-1]]
    frameIndex[i] = frameIndex[i][:-1]
    alignedVideoFrameTimes.append(frameTimes[i][:-1])
    
    for i,(videoType,sbin) in enumerate(zip(('side','face'),(1,2))):    
        likelihood = dlcData[str(sessionId)][videoType][dlcLabel]['likelihood'][()][frameIndex[i]]
        x,y = [int(np.average(dlcData[str(sessionId)][videoType][dlcLabel][c][()][frameIndex[i]],weights=likelihood)) for c in ('x','y')]
        roi = np.array([x + roiParams[videoType]['xoffset'], y + roiParams[videoType]['yoffset'],
                        roiParams[videoType]['width'], roiParams[videoType]['height']])
        
        videoPath = videoTable.loc[videoSessionIndex[i],videoType+'_video']
        facemapSavePath = os.path.join(baseDir,'facemapOutput')
        
        t = time.perf_counter()
        
        proc = {'sx': np.array([0]),
                'sy': np.array([0]),
                'sbin': sbin,
                'fullSVD': True,
                'save_mat': True,
                'rois': [{'rind': 1,
                          'rtype': 'motion SVD',
                          'ivid': 0,
                          'xrange': np.arange(roi[0],roi[0]+roi[2]),
                          'yrange': np.arange(roi[1],roi[1]+roi[3])}],
                'savepath': facemapSavePath}
        
        facemap.process.run(filenames=[[videoPath]],
                            sbin=sbin,
                            motSVD=True,
                            movSVD=True,
                            GUIobject=None,
                            parent=None,
                            proc=proc,
                            savepath=facemapSavePath)
        
        print(time.perf_counter()-t)
dlcData.close()
# save alignedVideoFrameTimes

















