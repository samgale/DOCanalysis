# -*- coding: utf-8 -*-
"""
Created on Thu Oct 20 10:35:04 2022

@author: svc_ccg
"""

import os
from simple_slurm import Slurm
from allensdk.brain_observatory.behavior.behavior_project_cache.behavior_neuropixels_project_cache import VisualBehaviorNeuropixelsProjectCache


# script to run
script_path = os.path.join(os.path.expanduser('~'),'PythonScripts','maskTaskModelHPC.py')
print(f'running {script_path}')

# define the job record output folder
stdout_location = os.path.join(os.path.expanduser('~'),'job_records')

# make the job record location if it doesn't already exist
os.mkdir(stdout_location) if not os.path.exists(stdout_location) else None

# build the python path
conda_environment = 'vbn'
python_path = os.path.join(os.path.expanduser('~'), 
                           'miniconda3', 
                           'envs', 
                           conda_environment,
                           'bin',
                           'python')

slurm = Slurm(cpus_per_task=1,
              partition='braintv',
              job_name='vbn session metrics',
              output=f'{stdout_location}/{Slurm.JOB_ARRAY_MASTER_ID}_{Slurm.JOB_ARRAY_ID}.out',
              time='2:00:00',
              mem_per_cpu='32gb')

vbn_cache = r'/allen/aibs/informatics/chris.morrison/ticket-27/allensdk_caches/vbn_cache_2022_Jul29/'
cache = VisualBehaviorNeuropixelsProjectCache.from_local_cache(cache_dir=vbn_cache)
ecephys_sessions_table = cache.get_ecephys_session_table(filter_abnormalities=False)

# call the `sbatch` command to run the jobs
for session, row in ecephys_sessions_table.iterrows():
    slurm.sbatch('{} {} --session_id {} --cache_dir {}'.format(
                 python_path,
                 script_path,
                 session,
                 vbn_cache))
