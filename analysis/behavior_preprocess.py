#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 12:42:16 2026

@author: yatangli
"""

import h5py
import numpy as np
import pandas as pd
from datetime import datetime
import os, sys
from scipy.stats import norm
import matplotlib.pyplot as plt


abspath = os.path.abspath(__file__)
current_dir = os.path.dirname(abspath)
parent_dir = os.path.dirname(current_dir)
os.chdir(parent_dir)
if parent_dir not in sys.path: 
    sys.path.append(parent_dir)
from utils.behavior_utils import *

def csv_raw_info(h5_path,save_path):
    whole_process_summary = []
    with h5py.File(h5_path,'r') as f:
        mouseid = f.attrs.get('mouse_id')
        record_group = f['training_records']
        all_dates = list(record_group.keys())
        for date_str in all_dates:
            daily_summary = {}
            dprimes,summary = bin_trials_and_compute_dprime(h5_path,date_str=date_str)
            daily_summary['date'] = date_str
            daily_summary['state'] = 'condition'
            daily_summary['trial_number'] = summary['n_trials']
            daily_summary['dprime'] = summary['dprime']
            daily_summary['accuracy'] = summary['accuracy']
            daily_summary['hit_rate'] = summary['hit_rate']
            daily_summary['fa_rate'] = summary['fa_rate']
            for i in range(5):
                try:
                    daily_summary[f'dprime_bin{i}'] = dprimes[i]['dprime']
                    daily_summary[f'accuracy_bin{i}'] = dprimes[i]['accuracy']
                    daily_summary[f'hit_rate_bin{i}'] = dprimes[i]['hit_rate']
                    daily_summary[f'fa_rate_bin{i}'] = dprimes[i]['fa_rate']
                except:
                    daily_summary[f'dprime_bin{i}'] = np.nan
                    daily_summary[f'accuracy_bin{i}'] = np.nan
                    daily_summary[f'hit_rate_bin{i}'] = np.nan
                    daily_summary[f'fa_rate_bin{i}'] = np.nan
            whole_process_summary.append(daily_summary)
    df = pd.DataFrame(whole_process_summary)
    columns_order = ['date', 'state', 'trial_number', 'dprime', 'accuracy', 'hit_rate',
       'fa_rate', 'dprime_bin0','dprime_bin1', 'dprime_bin2','dprime_bin3','dprime_bin4',
       'accuracy_bin0', 'accuracy_bin1', 'accuracy_bin2', 'accuracy_bin3', 'accuracy_bin4', 
       'hit_rate_bin0','hit_rate_bin1','hit_rate_bin2','hit_rate_bin3','hit_rate_bin4',
       'fa_rate_bin0', 'fa_rate_bin1', 'fa_rate_bin2', 'fa_rate_bin3','fa_rate_bin4']
    df = df[columns_order]
    df = df.sort_values(['date']).reset_index(drop=True)
    save_csv = os.path.join(save_path, f"{mouseid}_raw_to_bin.csv")
    df.to_csv(save_csv,na_rep='NaN', index=False, float_format='%.4f')
    
    
    
    
def generate_lick_frequencies_csv(hdf5_path, save_dir=None):
    
    # 时间窗口定义
    time_windows = [
        (0, 1.25, "Get Ready"),
        (1.25, 2.25, "Pure Visual Stimuli"),
        (2.25, 4.25, "Response Window")
    ]
    
    all_data = []
    
    try:
        with h5py.File(hdf5_path, 'r') as f:
            mouse_id = f.attrs.get('mouse_id')
            if 'training_records' not in f:
                print(f"警告: 文件中没有'training_records'组")
                return pd.DataFrame()
            
            root = f['training_records']
            all_dates = sorted(root.keys())
            
            for date in all_dates:
                date_str = str(date)
                date_group = root[date]
                sessions = sorted(date_group.keys())
                for session_name in sessions:
                    session_group = date_group[session_name]
                    session_trials = []
                    trial_types = ["Hit", "Miss", "FA", "CR"]
                    for trial_type in trial_types:
                        case_key = f"{trial_type}_licks"
                        if case_key in session_group:
                            try:
                                licks_data = session_group[case_key][()]
                                if licks_data.size == 0:
                                    continue
                                if len(licks_data.shape) != 2:
                                    warnings.warn(f"警告: {date_str}/{session_name}/{case_key}数据形状异常: {licks_data.shape}")
                                    continue
                                if licks_data.shape[0] == 0:
                                    continue
                                for i in range(licks_data.shape[0]):
                                    trial_index = int(licks_data[i, 0])
                                    timestamps = licks_data[i, 1:]
                                    timestamps = timestamps[~np.isnan(timestamps)] 
                                    lick_freqs = []
                                    for start, end, _ in time_windows:
                                        window_length = end - start
                                        lick_count = np.sum((timestamps >= start) & (timestamps <= end))
                                        if window_length > 0:
                                            lick_freq = lick_count / window_length
                                        else:
                                            lick_freq = 0   
                                        lick_freqs.append(lick_freq)
                                    
                                    # 添加到session_trials列表
                                    session_trials.append({
                                        'date': date_str,
                                        'session': session_name,
                                        'trial_index': trial_index,
                                        'trial_type': trial_type,
                                        'lick_freq_1': lick_freqs[0],
                                        'lick_freq_2': lick_freqs[1],
                                        'lick_freq_3': lick_freqs[2]
                                    })
                                
                            except Exception as e:
                                print(f"    错误: 处理{date_str}/{session_name}/{case_key}时出错: {e}")
                                continue
                    
                    if session_trials:
                        session_trials.sort(key=lambda x: x['trial_index'])

                        for trial in session_trials:
                            #trial['mouse_id'] = mouse_id
                            all_data.append(trial)
                    
    
    except Exception as e:
        print(f"处理文件 {hdf5_path} 时出错: {e}")
        return pd.DataFrame()
    if not all_data:
        print("警告: 没有找到任何数据！")
        return pd.DataFrame()
    
    df = pd.DataFrame(all_data)
    
    column_order = ['date', 'session', 'trial_index', 'trial_type', 
                    'lick_freq_1', 'lick_freq_2', 'lick_freq_3']
    df = df[column_order]
    
    df = df.sort_values(['date', 'session', 'trial_index']).reset_index(drop=True)
    
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        csv_filename = f"{mouse_id}_lick_frequency_per_trial.csv"
        csv_path = os.path.join(save_dir, csv_filename)
        df.to_csv(csv_path, index=False)
        print(f"CSV文件已保存: {csv_path}")
    else:
        csv_filename = f"{mouse_id}_lick_frequency_per_trial.csv"
        df.to_csv(csv_filename, index=False)
        print(f"CSV文件已保存: {csv_filename}")
    return df


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
    
    
# save_dir = os.path.join(base_dir, mouse_id,'behavioral_results')
csv_path = os.path.join(save_dir,f'{mouse_id}_summary_w_lick_frequency.csv')
if not os.path.exists(csv_path):
    
    
plot_individual_metrics_from_csv(mouseid=mouse_id,csv_path=csv_path,save_dir=save_dir)