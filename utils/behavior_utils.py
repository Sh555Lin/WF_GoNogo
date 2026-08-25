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
import warnings


abspath = os.path.abspath(__file__)
current_dir = os.path.dirname(abspath)
parent_dir = os.path.dirname(current_dir)
os.chdir(parent_dir)
if parent_dir not in sys.path: 
    sys.path.append(parent_dir)

def calculate_dprime(hit_rate, fa_rate):
    hit_rate = min(max(hit_rate, 0.01), 0.99)
    fa_rate = min(max(fa_rate, 0.01), 0.99)
    return norm.ppf(hit_rate) - norm.ppf(fa_rate)

def compute_sliding_bins(n_trials, step=50, bin_size=100, min_last_bin_size=91):
    bin_ranges = []
    start = 0
    while start + bin_size <= n_trials:
        bin_ranges.append((start, start + bin_size))
        start += step
    # 尾巴 bin：剩下的长度大于阈值，且不重叠添加
    if start < n_trials and (n_trials - start) >= min_last_bin_size:
        # 但避免 bin 长度为 0 的问题
        bin_ranges.append((start, n_trials))
    return bin_ranges

def bin_trials_and_compute_dprime(hdf5_path, date_str):
    all_trials = []

    with h5py.File(hdf5_path, 'r') as f:
        if 'training_records' not in f:
            raise ValueError("HDF5 中不包含 'training_records' 组")
        
        record_group = f['training_records']
        if date_str not in record_group:
            raise ValueError(f"日期 `{date_str}` 不在 `training_records` 中，请检查拼写或格式（应为 YYYYMMDD）")

        date_group = record_group[date_str]
        sessions = sorted(date_group.keys())

        for sess_key in sessions:
            trial_result = date_group[sess_key]["trial_results"][:]
            all_trials.append(trial_result)

    # 拼接所有 trial
    trial_seq = np.concatenate(all_trials)
    n_trials = len(trial_seq)

    # bin 划分
    bin_ranges = compute_sliding_bins(n_trials, step=50, bin_size=100, min_last_bin_size=91)

    dprimes = []
    for start, end in bin_ranges:
        bin_data = trial_seq[start:end]
        n_hit = np.sum(bin_data == 1)
        n_miss = np.sum(bin_data == 2)
        n_fa = np.sum(bin_data == 3)
        n_cr = np.sum(bin_data == 4)

        hit_total = n_hit + n_miss
        fa_total = n_fa + n_cr

        hit_rate = n_hit / hit_total if hit_total > 0 else 0.5
        fa_rate = n_fa / fa_total if fa_total > 0 else 0.5
        accuracy = (n_hit+n_cr)/bin_data.size

        d = calculate_dprime(hit_rate, fa_rate)

        # 文本 bin range 从 1 起数
        bin_label = f"{start+1}-{end}"

        dprimes.append({
            "bin_range": bin_label,
            "start": start,
            "end": end,
            "n_trials": end - start,
            "hit_rate": hit_rate,
            "fa_rate": fa_rate,
            "dprime": d,
            "accuracy":accuracy
        })

    # 全天整体数据
    n_hit_all = np.sum(trial_seq == 1)
    n_miss_all = np.sum(trial_seq == 2)
    n_fa_all = np.sum(trial_seq == 3)
    n_cr_all = np.sum(trial_seq == 4)

    hit_total_all = n_hit_all + n_miss_all
    fa_total_all = n_fa_all + n_cr_all

    hit_rate_all = n_hit_all / hit_total_all if hit_total_all > 0 else 0.5
    fa_rate_all = n_fa_all / fa_total_all if fa_total_all > 0 else 0.5
    dprime_all = calculate_dprime(hit_rate_all, fa_rate_all)
    accuracy_all = (n_hit_all+n_cr_all)/len(trial_seq)
    summary = {
        "hit_rate": hit_rate_all,
        "fa_rate": fa_rate_all,
        "dprime": dprime_all,
        "n_trials": n_trials,
        "accuracy":accuracy_all}

    if len(trial_seq)>200:
        trial_seq_200 = trial_seq[:200]
        n_hit_200 = np.sum(trial_seq_200 == 1)
        n_miss_200 = np.sum(trial_seq_200 == 2)
        n_fa_200 = np.sum(trial_seq_200 == 3)
        n_cr_200 = np.sum(trial_seq_200 == 4)

        hit_total_200 = n_hit_200 + n_miss_200
        fa_total_200 = n_fa_200 + n_cr_200

        hit_rate_200 = n_hit_200 / hit_total_200 if hit_total_200> 0 else 0.5
        fa_rate_200 = n_fa_200 / fa_total_200 if fa_total_200 > 0 else 0.5
        dprime_200 = calculate_dprime(hit_rate_200, fa_rate_200)
        accuracy_200 = (n_hit_200+n_cr_200)/200
        summary.update({
            "hit_rate_200": hit_rate_200,
            "fa_rate_200": fa_rate_200,
            "dprime_200": dprime_200,
            "accuracy_200":accuracy_200
        })

    return dprimes, summary

def calculate_date_dprime(hdf5_path, date_str):
    """
    按日期合并所有 session 的 trial_result，计算全天的 d'、hit rate 和 FA rate
    data_str的格式需要注意
    """
    all_trials = []

    with h5py.File(hdf5_path, 'r') as f:
        if 'training_records' not in f:
            raise ValueError("HDF5 中不包含 'training_records' 组")
        
        record_group = f['training_records']
        if date_str not in record_group:
            raise ValueError(f"日期 `{date_str}` 不在 `training_records` 中，请检查拼写或格式（应为 YYYYMMDD）")

        date_group = record_group[date_str]
        sessions = sorted(date_group.keys())

        for sess_key in sessions:
            trial_result = date_group[sess_key]["trial_results"][:]
            all_trials.append((sess_key, trial_result))

    # 拼接所有 trial
    sorted_trials = [trials for _, trials in sorted(all_trials)]
    trial_seq = np.concatenate(sorted_trials)

    # --- 全天整体 d', hit rate, FA rate ---
    n_hit_all = np.sum(trial_seq == 1)
    n_miss_all = np.sum(trial_seq == 2)
    n_fa_all = np.sum(trial_seq == 3)
    n_cr_all = np.sum(trial_seq == 4)

    hit_total_all = n_hit_all + n_miss_all
    fa_total_all = n_fa_all + n_cr_all

    hit_rate_all = n_hit_all / hit_total_all if hit_total_all > 0 else 0.5
    fa_rate_all = n_fa_all / fa_total_all if fa_total_all > 0 else 0.5
    dprime_all = calculate_dprime(hit_rate_all, fa_rate_all)
    accuracy_all = (n_hit_all+n_cr_all)/len(trial_seq)
    
    summary = {
        "hit_rate": hit_rate_all,
        "fa_rate": fa_rate_all,
        "dprime": dprime_all,
        "accuracy":accuracy_all,
        "n_trials": len(trial_seq)
    }

    return summary

def read_dprime_bins(dprime_bins):
    for b in dprime_bins:
        print(f"{b['start']+1:>4}-{b['end']:>4} | d': {b['dprime']:.2f} | hit_rate: {b['hit_rate']:.2f} | fa_rate: {b['fa_rate']:.2f}")

def compute_lick_frequency_vector(timestamps,fa,timeout):
    #fa=True False,
    #timeout = float
    """
    计算一个 trial 的舔水频率向量 (x, y, z, k)

    x: pre-stimulus window [0, 1.4]
    y: delay window [1.4, 2.4]
    z: response window [2.4, 4.4]
    k: interval window [4.4, 5.5]
    """
    x = np.sum((timestamps >= 0) & (timestamps <= 1.25)) / 1.25
    y = np.sum((timestamps > 1.25) & (timestamps <= 2.25)) / 1.0
    z = np.sum((timestamps > 2.25) & (timestamps <= 4.25)) / 2.0
    if not fa:
        k = np.sum((timestamps > 4.25) & (timestamps <= 5.5)) / 1.0
    else:
        k = np.sum((timestamps > 4.25) & (timestamps <= 5.5)) / timeout
    return np.array([x, y, z, k])

def lick_vectors(hdf5_path, case, date_list, timeout):
    case_key = f"{case}_licks"
    count_key = "trial_results"
    case_dic = {'Hit': 1, 'Miss': 2, 'FA': 3, 'CR': 4}
    if case=='FA':
        fa=True
    else:
        fa=False

    all_date_vectors = {}  # {date: [4D vectors]}
    
    with h5py.File(hdf5_path, 'r') as f:
        root = f['training_records']
        
        for date in date_list:
            if date not in root:
                continue
            date_group = root[date]
            vectors = []

            for session_name in date_group:
                session_group = date_group[session_name]
                if case_key in session_group:
                    licks = session_group[case_key][()]  # shape: (n_trials, n_licks + 1)
                    licks = licks[:, 1:]  # 去掉 trial label 列
                    trial_result = session_group[count_key][()]

                    for trial_idx, row in enumerate(licks):
                        #if trial_result[trial_idx] != case_dic[case]:
                        #    continue
                        timestamps = row[~np.isnan(row)]
                        if len(timestamps) == 0:
                            continue
                        vector = compute_lick_frequency_vector(timestamps,fa=fa,timeout=timeout)
                        vectors.append(vector)

            if vectors:
                all_date_vectors[date] = np.vstack(vectors)

    if not all_date_vectors:
        print("未找到任何有效的数据。")
        return

    return all_date_vectors  # dict[date] = n_trials x 4 array



def csv_raw_info(h5_path,save_path,mouse_id):
    whole_process_summary = []
    with h5py.File(h5_path,'r') as f:
        # mouse_id = f.attrs.get('mouse_id')
        # if mouse_id is None:
        #     mouse_id = 
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
    save_csv = os.path.join(save_path, f"{mouse_id}_raw_to_bin.csv")
    df.to_csv(save_csv,na_rep='NaN', index=False, float_format='%.4f')
    
    
    
    
def generate_lick_frequencies_csv(hdf5_path, save_dir=None, mouse_id=None):
    
    # 时间窗口定义
    time_windows = [
        (0, 1.25, "Get Ready"),
        (1.25, 2.25, "Pure Visual Stimuli"),
        (2.25, 4.25, "Response Window")
    ]
    
    all_data = []
    
    try:
        with h5py.File(hdf5_path, 'r') as f:
            # mouse_id = f.attrs.get('mouse_id')
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
    return df,csv_filename


def compute_sliding_bins(n_trials, step=50, bin_size=100, min_last_bin_size=91):
    bin_ranges = []
    start = 0
    while start + bin_size <= n_trials:
        bin_ranges.append((start, start + bin_size))
        start += step
    # 尾巴 bin：剩下的长度大于阈值，且不重叠添加
    if start < n_trials and (n_trials - start) >= min_last_bin_size:
        # 但避免 bin 长度为 0 的问题
        bin_ranges.append((start, n_trials))
    return bin_ranges

def calculate_dprime(hit_rate, fa_rate):
    hit_rate = min(max(hit_rate, 0.01), 0.99)
    fa_rate = min(max(fa_rate, 0.01), 0.99)
    return norm.ppf(hit_rate) - norm.ppf(fa_rate)

def compute_daily_summary(hdf5_path, date_str, step=10):
    """计算单日 dprime/accuracy，如果最后一个 bin hit_rate < 0.5，则使用 0.95~0.9 区间最大 dprime 替代"""
    all_trials = []

    with h5py.File(hdf5_path, 'r') as f:
        if 'training_records' not in f:
            raise ValueError("HDF5 中不包含 'training_records' 组")
        
        record_group = f['training_records']
        if date_str not in record_group:
            raise ValueError(f"日期 `{date_str}` 不在 `training_records` 中")

        date_group = record_group[date_str]
        sessions = sorted(date_group.keys())

        for sess_key in sessions:
            trial_result = date_group[sess_key]["trial_results"][:]
            all_trials.append(trial_result)

    trial_seq = np.concatenate(all_trials)
    n_trials = len(trial_seq)

    # --- 先计算 bin ---
    bin_ranges = compute_sliding_bins(n_trials, step=50, bin_size=100, min_last_bin_size=91)
    bin_hit_rates = []
    for start, end in bin_ranges:
        bin_data = trial_seq[start:end]
        n_hit = np.sum(bin_data == 1)
        n_miss = np.sum(bin_data == 2)
        hit_total = n_hit + n_miss
        hit_rate = n_hit / hit_total if hit_total > 0 else 0.5
        bin_hit_rates.append(hit_rate)

    # --- 全天整体数据 ---
    n_hit_all = np.sum(trial_seq == 1)
    n_miss_all = np.sum(trial_seq == 2)
    n_fa_all = np.sum(trial_seq == 3)
    n_cr_all = np.sum(trial_seq == 4)

    hit_total_all = n_hit_all + n_miss_all
    fa_total_all = n_fa_all + n_cr_all

    hit_rate_all = n_hit_all / hit_total_all if hit_total_all > 0 else 0.5
    fa_rate_all = n_fa_all / fa_total_all if fa_total_all > 0 else 0.5
    dprime_all = calculate_dprime(hit_rate_all, fa_rate_all)
    accuracy_all = (n_hit_all + n_cr_all) / n_trials

    summary = {
        "date": date_str,
        "n_trials": n_trials,
        "hit_rate": hit_rate_all,
        "fa_rate": fa_rate_all,
        "dprime": dprime_all,
        "accuracy": accuracy_all
    }
    #print(bin_hit_rates[-1])
    # --- 如果最后一个 bin 的 hit rate < 0.5，则用 0.95~0.9 的最大 dprime 替代 ---
    if bin_hit_rates and bin_hit_rates[-1] < 0.7 and n_trials > 200:
        best_dprime, best_acc = None, None
        #print('flag')
        for n in range(200, n_trials + 1, step):
            subseq = trial_seq[:n]
            n_hit = np.sum(subseq == 1)
            n_miss = np.sum(subseq == 2)
            n_fa = np.sum(subseq == 3)
            n_cr = np.sum(subseq == 4)

            hit_total = n_hit + n_miss
            fa_total = n_fa + n_cr

            hit_rate = n_hit / hit_total if hit_total > 0 else 0.5
            if hit_rate < 0.9 or hit_rate >= 0.99:  # 只考虑 [0.9, 0.95)
                continue
            fa_rate = n_fa / fa_total if fa_total > 0 else 0.5
            d = calculate_dprime(hit_rate, fa_rate)
            acc = (n_hit + n_cr) / n
            #print(d)
            if (best_dprime is None) or (d > best_dprime):
                #print('flag')
                best_dprime, best_acc = d, acc
                n_trials = n
        
        if best_dprime is not None:
            summary["dprime"] = best_dprime
            summary["accuracy"] = best_acc
            summary['n_trials'] = n_trials
            summary["hit_rate"] = hit_rate
            summary["fa_rate"] = fa_rate
    return summary

def compute_daily_summary_tocsv(hdf5_path,lick_frequency_csv,save_dir=None,mouse_id=None):
    
    all_summaries = []
    with h5py.File(hdf5_path, 'r') as f:
        # mouse_id = f.attrs.get('mouse_id')
        record_group = f['training_records']
        all_dates = sorted(record_group.keys())              
        for date_str in all_dates:
            state = 'Condition'
            summary = compute_daily_summary(hdf5_path, date_str)
            summary["date"] = date_str
            summary["state"] = state
            summary['miss_rate'] = 1-summary['hit_rate']
            summary['cr_rate'] = 1-summary['fa_rate']
                        
            all_summaries.append(summary)
                        
    
    summary_df = pd.DataFrame(all_summaries)
    column_order = ['date', 'state', 'n_trials', 'accuracy', 'dprime','hit_rate', 'fa_rate','miss_rate','cr_rate']
    summary_df = summary_df[column_order]
    summary_df = summary_df.sort_values(['date']).reset_index(drop=True)

    lick_df = pd.read_csv(lick_frequency_csv)
    lick_df['date'] = lick_df['date'].astype(str)
    #print(lick_df)
    lick_columns = [
        'hit_lick_1', 'hit_lick_2', 'hit_lick_3',
        'fa_lick_1', 'fa_lick_2', 'fa_lick_3',
        'miss_lick_1', 'miss_lick_2', 'miss_lick_3',
        'cr_lick_1', 'cr_lick_2', 'cr_lick_3',
        'total_lick_1', 'total_lick_2', 'total_lick_3'
    ]
    for col in lick_columns:
        summary_df[col] = np.nan

    for idx, row in summary_df.iterrows():
        date = str(row['date'])
        mask = lick_df['date']==date
        #print(mask)
        daily_lick_data = lick_df[mask].copy()

        hit_count = 0
        fa_count = 0
        miss_count = 0
        cr_count = 0
        
        hit_sum = [0.0, 0.0, 0.0]
        fa_sum = [0.0, 0.0, 0.0]
        miss_sum = [0.0, 0.0, 0.0]
        cr_sum = [0.0, 0.0, 0.0]
        total_sum = [0.0, 0.0, 0.0]
        for _, trial in daily_lick_data.iterrows():
            trial_type = trial['trial_type']
            
            # 获取三个窗口的舔舐频率
            lick_vals = [
                trial['lick_freq_1'],
                trial['lick_freq_2'],
                trial['lick_freq_3']
            ]
            
            # 累加到总和中
            for i in range(3):
                total_sum[i] += lick_vals[i]
            
            # 根据trial类型分类累加
            if trial_type == 'Hit':
                hit_count += 1
                for i in range(3):
                    hit_sum[i] += lick_vals[i]
            elif trial_type == 'FA':
                fa_count += 1
                for i in range(3):
                    fa_sum[i] += lick_vals[i]
            elif trial_type == 'Miss':
                miss_count += 1
                for i in range(3):
                    miss_sum[i] += lick_vals[i]
            elif trial_type == 'CR':
                cr_count += 1
                for i in range(3):
                    cr_sum[i] += lick_vals[i]
        for i in range(3):
            if hit_count > 0:
                summary_df.at[idx, f'hit_lick_{i+1}'] = hit_sum[i] / hit_count
            else:
                summary_df.at[idx, f'hit_lick_{i+1}'] = np.nan
        
        # FA舔舐频率
        for i in range(3):
            if fa_count > 0:
                summary_df.at[idx, f'fa_lick_{i+1}'] = fa_sum[i] / fa_count
            else:
                summary_df.at[idx, f'fa_lick_{i+1}'] = np.nan
        
        # Miss舔舐频率
        for i in range(3):
            if miss_count > 0:
                summary_df.at[idx, f'miss_lick_{i+1}'] = miss_sum[i] / miss_count
            else:
                summary_df.at[idx, f'miss_lick_{i+1}'] = np.nan
        
        # CR舔舐频率
        for i in range(3):
            if cr_count > 0:
                summary_df.at[idx, f'cr_lick_{i+1}'] = cr_sum[i] / cr_count
            else:
                summary_df.at[idx, f'cr_lick_{i+1}'] = np.nan
        
        # 总体舔舐频率（所有前n_trials个trial）
        total_trials = len(daily_lick_data)
        for i in range(3):
            if total_trials > 0:
                summary_df.at[idx, f'total_lick_{i+1}'] = total_sum[i] / total_trials
            else:
                summary_df.at[idx, f'total_lick_{i+1}'] = np.nan

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        csv_filename = f"{mouse_id}_summary_w_lick_frequency.csv"
        csv_path = os.path.join(save_dir, csv_filename)
        summary_df.to_csv(csv_path, index=False)
        print(f"CSV文件已保存: {csv_path}")
    else:
        csv_filename = f"{mouse_id}_summary_w_lick_frequency.csv"
        summary_df.to_csv(csv_filename, index=False)
        print(f"CSV文件已保存: {csv_filename}")

    return summary_df

def plot_individual_metrics_from_csv(mouse_id, csv_path, save_dir):
    df = pd.read_csv(csv_path)
    
    # 确保日期列是日期格式
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%d')
    
    metrics = ['dprime', 'accuracy', 'hit_rate', 'fa_rate']
    titles = ['d-prime', 'Accuracy', 'Hit Rate', 'False Alarm Rate']
    y_labels = ['d-prime', 'Accuracy', 'Hit Rate', 'False Alarm Rate']
    
    states = ['Condition', 'Discrimination', 'Switch']
    state_colors = {'Condition': '#FFE5E5', 'Discrimination': '#E5F7FF', 'Switch': '#F0FFE5'}
    
    # 按日期排序
    df = df.sort_values('date')
    # 添加连续索引，忽略缺失的日期
    df['session_index'] = range(len(df))
    
    for metric, title, y_label in zip(metrics, titles, y_labels):
        plt.figure(figsize=(max(15, len(df) * 0.7), 6))  # 调整宽度
        ax = plt.gca()
        
        # 按状态分组绘制，使用连续索引作为x轴
        for state in states:
            if state in df['state'].unique():
                state_data = df[df['state'] == state]
                state_x = state_data['session_index'].values  # 使用连续索引
                state_y = state_data[metric].values
                
                if len(state_x) > 1:
                    # 绘制连续线
                    ax.plot(state_x, state_y, 'o-', linewidth=2, markersize=6)
                elif len(state_x) == 1:
                    # 单个点
                    ax.plot(state_x, state_y, 'o', markersize=8)
        
        # 添加状态背景色，使用连续索引
        for state in states:
            if state in df['state'].unique():
                state_data = df[df['state'] == state]
                if len(state_data) > 0:
                    start_idx = state_data['session_index'].min()
                    end_idx = state_data['session_index'].max()
                    
                    # 稍微扩展背景范围以获得更好的视觉效果
                    ax.axvspan(start_idx - 0.4, end_idx + 0.4, alpha=0.2, 
                              color=state_colors[state], zorder=-1)
        
        # 设置 y 轴限制
        if metric in ['hit_rate', 'fa_rate', 'miss_rate', 'cr_rate', 'accuracy']:
            ax.set_ylim([-0.05, 1.1])
        
        ax.set_title(f'{mouse_id} {title} Across Training States', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('Training Sessions (Date)', fontsize=14)
        ax.set_ylabel(y_label, fontsize=14)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # 设置 x 轴刻度：每个数据点一个刻度
        ax.set_xticks(df['session_index'].values)
        
        # 格式化日期显示
        date_labels = [date.strftime('%Y-%m-%d') for date in df['date']]
        ax.set_xticklabels(date_labels, rotation=45, ha='right', fontsize=10)
        
        # 根据数据点数量调整标签显示
        if len(df) > 40:
            # 如果数据点很多，可以间隔显示日期标签
            rotation_angle = 60
            ha_align = 'right'
            fontsize = 9
            
            # 选择要显示的刻度（每N个显示一个）
            if len(df) > 60:
                step = max(1, len(df) // 15)  # 大约显示15个标签
                visible_indices = list(range(0, len(df), step))
                # 确保显示第一个和最后一个
                if visible_indices[-1] != len(df) - 1:
                    visible_indices.append(len(df) - 1)
                
                ax.set_xticks(df.loc[visible_indices, 'session_index'].values)
                visible_labels = [date_labels[i] for i in visible_indices]
                ax.set_xticklabels(visible_labels, rotation=rotation_angle, 
                                  ha=ha_align, fontsize=fontsize)
            else:
                plt.xticks(rotation=rotation_angle, ha=ha_align, fontsize=fontsize)
        elif len(df) > 25:
            rotation_angle = 45
            ha_align = 'right'
            fontsize = 9
            plt.xticks(rotation=rotation_angle, ha=ha_align, fontsize=fontsize)
        else:
            plt.xticks(rotation=45, ha='right', fontsize=10)
        
        # 添加状态分隔线
        state_changes = df['state'].ne(df['state'].shift()).where(df['state'].ne(df['state'].shift())).dropna()
        for idx in state_changes.index:
            if idx > 0:
                # 分隔线在两个状态之间
                x_val = df.loc[idx, 'session_index'] - 0.5
                ax.axvline(x=x_val, color='gray', linestyle='-', 
                          linewidth=1, alpha=0.7)
        
        # 在 x 轴刻度处添加细网格线
        ax.xaxis.grid(True, which='major', linestyle=':', alpha=0.3, linewidth=0.5)
        
        # 添加日期范围信息到图例或图表
        date_range_str = f"Date Range: {df['date'].min().strftime('%Y-%m-%d')} to {df['date'].max().strftime('%Y-%m-%d')}"
        ax.text(0.02, 0.98, date_range_str, transform=ax.transAxes, 
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 创建状态图例
        state_handles = [plt.Rectangle((0,0), 1, 1, color=state_colors[state], alpha=0.2) 
                         for state in states if state in df['state'].unique()]
        state_labels = [state for state in states if state in df['state'].unique()]
        
        ax.legend(state_handles, state_labels, title='State', 
                  loc='upper left', bbox_to_anchor=(1.02, 0.6), 
                  borderaxespad=0., fontsize=10, frameon=True, 
                  fancybox=True, shadow=True)
        
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'{metric}_of_{mouse_id}_by_session.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.savefig(f'{metric}_of_{mouse_id}_by_session.png', dpi=300, bbox_inches='tight')
            print(f"Plot saved to {metric}_by_session.png")
        
        plt.show()
#%%

