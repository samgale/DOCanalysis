# -*- coding: utf-8 -*-
"""
Created on Wed Sep 13 15:25:55 2023

@author: svc_ccg
"""

import os
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['pdf.fonttype'] = 42
import cv2



baseDir = r"C:\Users\svc_ccg\Desktop\Analysis\vbn"

stimTable = pd.read_csv(os.path.join(baseDir,'master_stim_table.csv'))

videoTable = pd.read_excel(os.path.join(baseDir,'vbn_video_paths_full_validation.xlsx'))
videoTable.insert(0,'session_id',[int(s[:s.find('_')]) for s in videoTable['exp_id']])

sessionIds = stimTable['session_id'].unique()


roi = {'NP.0': {'side': np.array([100,155,120,75]), 'face': np.array([205,100,170,180])},
       'NP.1': {'side': np.array([140,140,110,75]), 'face': np.array([220,100,170,180])}}

dlcLabel = 'nose_tip'
roiParams = {'side': {'xoffset': -10, 'yoffset': -60, 'width': 450, 'height': 200},
             'face': {'xoffset': -130, 'yoffset': -160, 'width': 450, 'height': 225}}


for sessionIndex,sessionId in enumerate(sessionIds):
    print(sessionIndex)
    stim = stimTable[(stimTable['session_id']==sessionId) & stimTable['active']].reset_index()
    stimStart = stim['start_time'].iloc[0]
    stimEnd = stim['stop_time'].iloc[-1]
    
    for videoType in ('side','face'):
        i = np.where(videoTable['session_id'] == sessionId)[0][0]
        rigId = videoTable.loc[i,'rig_id']
        
        frameTimesPath = videoTable.loc[i,videoType+'_timestamp_path']
        frameTimes = np.load(frameTimesPath)
        frameIndex = np.where((frameTimes >= stimStart) & (frameTimes <= stimEnd))[0]
        
        dlcPath = videoTable.loc[i,videoType+'_dlc_output']
        dlcOutput = pd.read_hdf(dlcPath).droplevel('scorer',axis='columns')
        likelihood = dlcOutput[dlcLabel]['likelihood'][frameIndex]
        x,y = [int(np.average(dlcOutput[dlcLabel][c][frameIndex],weights=likelihood)) for c in ('x','y')]
        roi = np.array([x + roiParams[videoType]['xoffset'], y + roiParams[videoType]['yoffset'],
                        roiParams[videoType]['width'], roiParams[videoType]['height']])
        
        videoPath = videoTable.loc[i,videoType+'_video']
        video = cv2.VideoCapture(videoPath)
        video.set(cv2.CAP_PROP_POS_FRAMES,frameIndex[0])
        isImage,image = video.read()
        image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        image = cv2.rectangle(image,roi[:2],roi[:2]+roi[-2:],255,1)

        cv2.imwrite(os.path.join(baseDir,'videoROIs',str(sessionId)+'_'+videoType+'_roi.png'),image)
                    
        video.release()



baseDir = r"C:\Users\svc_ccg\Desktop\Analysis\vbn"

videoTable = pd.read_excel(os.path.join(baseDir,'vbn_video_paths_full_validation.xlsx'))
videoTable.insert(0,'session_id',[int(s[:s.find('_')]) for s in videoTable['exp_id']])

h5Path = r'C:/Users/svc_ccg/Desktop/Analysis/vbn/vbnDLCdata.hdf5'
h5File = h5py.File(h5Path,'w')

for sessionIndex,sessionId in enumerate(videoTable['session_id']):
    print(sessionIndex)
    sessionGroup = h5File.create_group(str(sessionId))
    for videoType in ('side','face','eye'):
        videoGroup = sessionGroup.create_group(videoType)
        i = np.where(videoTable['session_id'] == sessionId)[0][0] 
        dlcPath = videoTable.loc[i,videoType+'_dlc_output']
        dlc = pd.read_hdf(dlcPath).droplevel('scorer',axis='columns')
        labels = dlc.columns.get_level_values(0).unique()
        for lbl in labels:
            labelGroup = videoGroup.create_group(lbl)
            for key in ('likelihood','x','y'):
                labelGroup.create_dataset(key,data=dlc[lbl][key],compression='gzip',compression_opts=4)
                
h5File.close()     




















