#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  7 17:34:35 2026

@author: yatangli
"""

import h5py
import numpy as np
import pandas as pd
from datetime import datetime
import os, sys
from scipy.stats import norm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

abspath = os.path.abspath(__file__)
current_dir = os.path.dirname(abspath)
parent_dir = os.path.dirname(current_dir)
os.chdir(parent_dir)
if parent_dir not in sys.path: 
    sys.path.append(parent_dir)
from utils.behavior_utils import *

#%%
base_dir = "/Volumes/Data_attention/Transfer learning/LinShu/DATA_linshu/000 Widefield"
# mouse_id = "A095"
mouse_id = "A269"
# Assuming base_dir and mouse_id are already defined
old_save_dir = os.path.join(base_dir, mouse_id, 'results')
new_save_dir = os.path.join(base_dir, mouse_id, 'behavioral_results')
# Check if the original 'results' directory exists
if os.path.exists(new_save_dir):
    save_dir = new_save_dir
    # Create the new directory path
elif os.path.exists(old_save_dir):
    # Rename the directory
    os.rename(old_save_dir, new_save_dir)
    save_dir = new_save_dir
    
h5_path = os.path.join(save_dir,f'{mouse_id}_discrimination_data.h5')
# save_dir = os.path.join(base_dir, mouse_id,'behavioral_results')
csv_path = os.path.join(save_dir,f'{mouse_id}_summary_w_lick_frequency.csv')
if not os.path.exists(csv_path):
    csv_raw_info(h5_path,save_path=save_dir, mouse_id=mouse_id)
    _,lick_frequency_csv = generate_lick_frequencies_csv(h5_path, save_dir=save_dir,mouse_id=mouse_id)
    compute_daily_summary_tocsv(h5_path,lick_frequency_csv= os.path.join(save_dir,lick_frequency_csv), save_dir=save_dir,mouse_id=mouse_id)
    
plot_individual_metrics_from_csv(mouse_id=mouse_id,csv_path=csv_path,save_dir=save_dir)
