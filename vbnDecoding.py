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



baseDir = r"C:\Users\svc_ccg\Desktop\Analysis\vbn"

stimTable = pd.read_csv(os.path.join(baseDir,'master_stim_table.csv'))

videoTable = pd.read_excel(os.path.join(baseDir,'vbn_video_paths_full_validation.xlsx'))
videoTable.insert(0,'session_id',[int(s[:s.find('_')]) for s in videoTable['exp_id']])

sessionIds = stimTable['session_id'].unique()


for session in sessionIds:
    i = videoTable['session_id'] == session
    faceFrameTimes = np.load(videoTable.loc[i,'face_timestamp_path'].iloc[0])


























