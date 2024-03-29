# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 10:35:04 2022

@author: svc_ccg
"""

import os, glob
import numpy as np
import pandas as pd
from simple_slurm import Slurm

# script to run
script_path = '/allen/ai/homedirs/samg/PythonScripts/vbnAnalysisHPC.py'

# job record output folder
stdout_location = '/allen/ai/homedirs/samg/job_records'

# python path
python_path = '/allen/programs/mindscope/workgroups/np-behavior/VBN_video_analysis/miniconda/envs/facemap/bin/python'

# call the `sbatch` command to run the jobs
slurm = Slurm(cpus_per_task=1,
              partition='braintv',
              job_name='vbnFacemap',
              output=f'{stdout_location}/{Slurm.JOB_ARRAY_MASTER_ID}_{Slurm.JOB_ARRAY_ID}.out',
              time='12:00:00',
              mem_per_cpu='16gb')

baseDir = '/allen/programs/mindscope/workgroups/np-behavior/vbn_data_release/supplemental_tables'

stimTable = pd.read_csv(os.path.join(baseDir,'master_stim_table.csv'))
sessionIds = stimTable['session_id'].unique()
# sessionIds = stimTable['session_id'][stimTable['experience_level']=='Novel'].unique()
for sessionId in sessionIds:
    slurm.sbatch('{} {} --sessionId {}'.format(python_path,script_path,sessionId))

# clusterTable = pd.read_csv(os.path.join(baseDir,'unit_cluster_labels.csv'))
# regions = ('all','LGd','VISp','VISl','VISrl','VISal','VISpm','VISam','LP',
#            'MRN','MB','SC','APN','NOT','Hipp')
# clusters = ['all'] + ['cluster_'+str(c) for c in np.unique(clusterTable['cluster_labels']) + 1]
# for label in ('change','lick','hit'):
#     for region in regions:
#         for cluster in clusters:
#             slurm.sbatch('{} {} --label {} --region {} --cluster {}'.format(python_path,script_path,label,region,cluster))
