# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 16:22:30 2022

@author: svc_ccg
"""


from allensdk.brain_observatory.behavior.behavior_project_cache.behavior_neuropixels_project_cache import VisualBehaviorNeuropixelsProjectCache


vbn_cache = r'\\allen\aibs\informatics\chris.morrison\ticket-27\allensdk_caches\vbn_cache_2022_Jul29'
#vbn_cache = r'/allen/programs/mindscope/workgroups/np-behavior/vbn_data_release/vbn_s3_cache'

cache = VisualBehaviorNeuropixelsProjectCache.from_local_cache(cache_dir=vbn_cache)

ecephys_sessions_table = cache.get_ecephys_session_table(filter_abnormalities=False)




























