import os
import glob
import h5py
import yaml
import numpy as np
import pandas as pd
import scipy.io as sio
from scipy.stats import norm
from collections import defaultdict
from utils.io_behavior import load_bpod_mat_files, extract_session_time_from_filename, write_trial_results_to_hdf5_from_cfg


def process_behavior_session(mat_path):
    """Process a single .mat session, extract lick timestamps and trial outcomes"""
    try:
        data = sio.loadmat(mat_path, struct_as_record=False, squeeze_me=True)['SessionData']
    except Exception as e:
        print(f"Failed to load: {mat_path}\n{e}")
        return None

    n_trial = int(data.nTrials)
    outcome = np.full(n_trial + 1, np.nan)
    outcome[-1] = n_trial

    sheets = {'HIT': [], 'MISS': [], 'FA': [], 'CR': []}

    for x in range(n_trial):
        try:
            trial = data.RawEvents.Trial[x]
            licks = getattr(trial.Events, 'BNC1High', [])
        except Exception:
            licks = []

        for key in sheets:
            try:
                state = getattr(data.RawEvents.Trial[x].States, key)
                if not np.isnan(state[0]):
                    outcome[x] = {'HIT': 1, 'MISS': 2, 'FA': 3, 'CR': 4}[key]
                    sheets[key].append([x + 1] + licks.tolist())  # MATLAB indices start at 1
                    break
            except Exception:
                continue

    session_id = extract_session_time_from_filename(os.path.basename(mat_path))
    return session_id, outcome, sheets


def pad_nested_list(nested):
    """Convert nested list to a padded numpy array, similar to MATLAB's padNestedCell"""
    if not nested:
        return np.array([[np.nan]])
    max_len = max(len(x) for x in nested)
    padded = np.full((len(nested), max_len), np.nan)
    for i, row in enumerate(nested):
        padded[i, :len(row)] = row
    return padded

def write_lick_excel(sheets_dict, session_labels, save_path):
    """Save lick time tables to Excel with multiple sheets"""
    with pd.ExcelWriter(save_path, engine='openpyxl', mode='w') as writer:
        for label, session in zip(session_labels, sheets_dict):
            if session is None or np.isnan(session).all():
                continue
            df = pd.DataFrame(session)
            try:
                df.to_excel(writer, sheet_name=label[:31], index=False, header=False)
            except Exception as e:
                print(f"Error writing sheet {label}: {e}")

def process_behavior_mouse(base_dir, mouse_id, sessions=None):
    mat_files, directory_path = load_bpod_mat_files(base_dir, mouse_id, sessions)
    if not mat_files:
        print(f'No .mat files found in {directory_path}')
        return


    results_per_session = []
    session_labels = []
    all_sheets = defaultdict(list)

    for path in sorted(mat_files):
        print(f'Processing {path}')
        result = process_behavior_session(path)
        if result is None:
            continue
        session_id, outcome, sheets = result
        session_labels.append(session_id)
        results_per_session.append(outcome)

        for key in ['HIT', 'MISS', 'FA', 'CR']:
            all_sheets[key].append(pad_nested_list(sheets[key]))

    # Write trial_results.csv
    save_csv_path = os.path.join(directory_path, 'trial_results.csv')
    write_trial_result_csv(results_per_session, session_labels, save_csv_path)

    # Write Excel files for each outcome type
    for key in ['HIT', 'MISS', 'FA', 'CR']:
        save_xlsx = os.path.join(directory_path, f'when_lick_{key.lower()}_ALL.xlsx')
        write_lick_excel(all_sheets[key], session_labels, save_xlsx)

def write_trial_result_csv(result_list, labels, save_path):
    """Write trial outcomes to a CSV file"""
    max_len = max(len(r) for r in result_list)
    with open(save_path, 'w') as f:
        for label, row in zip(labels, result_list):
            padded = list(row) + [np.nan] * (max_len - len(row))
            row_str = ','.join(['NaN' if np.isnan(x) else str(int(x)) for x in padded])
            f.write(f"{label},{row_str}\n")

def load_config(config_path):
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Cannot find config file: {config_path}")
    with open(config_path, 'r') as file:
        cfg = yaml.safe_load(file)
    return cfg

def run_behavior_from_config(config_path):
    cfg = load_config(config_path)
    write_trial_results_to_hdf5_from_cfg(cfg)

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
