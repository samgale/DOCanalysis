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
import h5py
import decord
import skvideo
skvideo.setFFmpegPath(r"C:\Users\svc_ccg\Desktop\ffmpeg\bin") # run this before importing skvideo.io
import skvideo.io
import facemap


baseDir = r"C:\Users\svc_ccg\Desktop\Analysis\vbn"

stimTable = pd.read_csv(os.path.join(baseDir,'master_stim_table.csv'))

videoTable = pd.read_excel(os.path.join(baseDir,'vbn_video_paths_full_validation.xlsx'))
videoTable.insert(0,'session_id',[int(s[:s.find('_')]) for s in videoTable['exp_id']])

sessionIds = stimTable['session_id'].unique()


dlcLabel = 'nose_tip'
roiParams = {'side': {'xoffset': -10, 'yoffset': -60, 'width': 450, 'height': 200},
             'face': {'xoffset': -130, 'yoffset': -160, 'width': 450, 'height': 225}}


dlcData = h5py.File(os.path.join(baseDir,'dlcData.hdf5'))
alignedVideoFrameTimes = []
for sessionIndex,sessionId in enumerate(sessionIds[:10]):
    print(sessionIndex)
    stim = stimTable[(stimTable['session_id']==sessionId) & stimTable['active']].reset_index()
    stimStart = stim['start_time'].iloc[0]
    stimEnd = stim['stop_time'].iloc[-1]
    
    frameTimes = []
    frameIndex = []
    for videoType in ('side','face'):
        i = np.where(videoTable['session_id'] == sessionId)[0][0]
        frameTimesPath = videoTable.loc[i,videoType+'_timestamp_path']
        ft = np.load(frameTimesPath)
        fi = np.where((ft >= stimStart) & (ft <= stimEnd))[0]
        frameTimes.append(ft[fi])
        frameIndex.append(fi)
    i,j = (0,1) if frameIndex[0].size >= frameIndex[1].size else (1,0)
    frameIndex[j] = frameIndex[j][np.searchsorted(frameTimes[j],frameTimes[i])[:-1]]
    frameIndex[i] = frameIndex[i][:-1]
    alignedVideoFrameTimes.append(frameTimes[i][:-1])
    
    for i,videoType in enumerate(('side','face')):    
        likelihood = dlcData[str(sessionId)][videoType][dlcLabel]['likelihood'][()][frameIndex[i]]
        x,y = [int(np.average(dlcData[str(sessionId)][videoType][dlcLabel][c][()][frameIndex[i]],weights=likelihood)) for c in ('x','y')]
        roi = np.array([x + roiParams[videoType]['xoffset'], y + roiParams[videoType]['yoffset'],
                        roiParams[videoType]['width'], roiParams[videoType]['height']])
        
        t = time.perf_counter()
        videoInPath = videoTable.loc[i,videoType+'_video']
        videoIn = decord.VideoReader(videoInPath)
        frameRate = videoIn.get_avg_fps()
        videoOutPath = os.path.join(baseDir,'videos',str(sessionId)+'_'+videoType+'.mp4')
        videoOut = skvideo.io.FFmpegWriter(videoOutPath,
                                           inputdict={'-r':str(frameRate)},
                                           outputdict={'-r':str(frameRate),'-vcodec':'libx264','-crf':'17'})
        for frame in frameIndex[i]:
            image = videoIn[frame].asnumpy()
            videoOut.writeFrame(image[roi[1]:roi[1]+roi[3],roi[0]:roi[0]+roi[2]])
        videoIn.release()
        videoOut.close()
        print(time.perf_counter()-t)
dlcData.close()
# save alignedVideoFrameTimes


side = cv2.VideoCapture(r'C:\\Users\\svc_ccg\\Desktop\\Analysis\\vbn\\videos\\1044385384_side.mp4')
face = cv2.VideoCapture(r'C:\\Users\\svc_ccg\\Desktop\\Analysis\\vbn\\videos\\1044385384_face.mp4')
videoOutPath = os.path.join(baseDir,'videos',str(sessionId)+'_merged.mp4')
videoOut = skvideo.io.FFmpegWriter(videoOutPath,
                                   inputdict={'-r':str(frameRate)},
                                   outputdict={'-r':str(frameRate),'-vcodec':'libx264','-crf':'17'})
for _ in range(3600):
    frame = np.zeros((425,450),dtype=np.uint8)
    isImage,image = side.read()
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    frame[:200,:] = image
    isImage,image = face.read()
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    frame[200:,:] = image
    videoOut.writeFrame(frame)
videoOut.close()
side.release()
face.release()



facemap.process

















