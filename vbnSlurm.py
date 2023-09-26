# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 10:35:04 2022

@author: svc_ccg
"""

import os
import pandas as pd
from simple_slurm import Slurm


# script to run
script_path = os.path.join(os.path.expanduser('~'),'PythonScripts','vbnAnalysisHPC.py')
print(f'running {script_path}')

# define the job record output folder
stdout_location = os.path.join(os.path.expanduser('~'),'job_records')

# make the job record location if it doesn't already exist
os.mkdir(stdout_location) if not os.path.exists(stdout_location) else None

# python path
python_path = '/allen/programs/mindscope/workgroups/np-behavior/VBN_video_analysis/miniconda/envs/facemap/python.exe'

# call the `sbatch` command to run the jobs
slurm = Slurm(cpus_per_task=1,
              partition='braintv',
              job_name='vbn session metrics',
              output=f'{stdout_location}/{Slurm.JOB_ARRAY_MASTER_ID}_{Slurm.JOB_ARRAY_ID}.out',
              time='6:00:00',
              mem_per_cpu='32gb')

baseDir = '/allen/programs/mindscope/workgroups/np-behavior/vbn_data_release/supplemental_tables'
stimTable = pd.read_csv(os.path.join(baseDir,'master_stim_table.csv'))
sessionIds = stimTable['session_id'].unique()
for sessionId in sessionIds:
    slurm.sbatch('{} {} --sessionId {}'.format(python_path,script_path,sessionId))
