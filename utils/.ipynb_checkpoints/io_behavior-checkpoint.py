import os
import glob
import h5py
import numpy as np
import pandas as pd
from datetime import datetime


def load_bpod_mat_files(base_dir, mouse_id, sessions=None):
    mouse_path = os.path.join(base_dir, mouse_id)
    mat_paths = []
    if sessions:
        for sess in sessions:
            session_dir = os.path.join(mouse_path, sess)
            mat_paths.extend(glob.glob(os.path.join(session_dir, '**', '*.mat'), recursive=True))
    else:
        mat_paths = glob.glob(os.path.join(mouse_path, '**', '*.mat'), recursive=True)
    return mat_paths, mouse_path



def extract_session_time_from_filename(filename):
    import re
    match = re.search(r'(\d{8}_\d{6})_', filename)
    return match.group(1) if match else filename



def build_paths(cfg):
    base_data_dir = cfg['base_dir'] 
    mouse_id = cfg['mouse_id']

    results_dir = os.path.join(base_data_dir, mouse_id, 'results')

    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    trial_result_csv = os.path.join(base_data_dir, mouse_id, 'trial_results.csv')
    h5_path = os.path.join(results_dir, f'{mouse_id}_discrimination_data.h5')

    lick_labels = ['Hit', 'Miss', 'CR', 'FA']
    lick_paths = {label: os.path.join(base_data_dir, mouse_id, f'when_lick_{label}_ALL.xlsx')
                  for label in lick_labels}

    return results_dir, h5_path, trial_result_csv, lick_paths


def load_lick_sheets(path):
    xls = pd.read_excel(path, sheet_name=None)
    return {k: v.values for k, v in xls.items()}


def write_trial_results_to_hdf5_from_cfg(cfg):
    """
    从 config 中构建路径并写入 trial_results 和 lick events 到 HDF5 文件。
    参数：
        - cfg: dict, 包含 base_data_dir、base_analysis_dir、mouse_id、mode
    """
    results_dir, h5_path, trial_result_path, lick_paths = build_paths(cfg)

    trial_df = pd.read_csv(trial_result_path, header=None, index_col=0)
    trial_df.index = trial_df.index.astype(str)
    trial_df = trial_df.sort_index()
    training_dates = np.unique(pd.Index(trial_df.index).str.split('_').str[0])

    lick_data = {label: load_lick_sheets(path) for label, path in lick_paths.items()}

    with h5py.File(h5_path, 'a') as f:
        birth_date = datetime.strptime(cfg['birth_date'], "%Y-%m-%d")
        if 'training_records' not in f:
            f.create_group('training_records')
        training_grp = f['training_records']

        for index, row in trial_df.iterrows():
            date_key = index.split('_')[0]
            session_key = index
            training_date_obj = datetime.strptime(date_key, '%Y%m%d')

            trial_results = row.values.astype('float')
            last_valid_index = np.where(~np.isnan(trial_results))[0][-1]
            trial_results = trial_results[:last_valid_index]

            # 日期层级
            if date_key not in training_grp:
                print(f"\nWriting data on date: {date_key}")
                day_grp = training_grp.create_group(date_key)
                age_days = (training_date_obj - birth_date).days
                age_weeks = int(np.ceil(age_days / 7))
                day_grp.attrs["age_week"] = age_weeks
                day_grp.attrs["training_day_index"] = np.where(training_dates == date_key)[0][0] + 1
            else:
                day_grp = training_grp[date_key]

            print(f"Session: {session_key} is adding...")
            if session_key in day_grp:
                del day_grp[session_key]
            sess_grp = day_grp.create_group(session_key)
            sess_grp.create_dataset("trial_results", shape=(trial_results.size,), data=trial_results)
            print("trial_results...done")

            # 写入lick
            for outcome in ["Hit", "Miss", "CR", "FA"]:
                ts_array = lick_data[outcome].get(session_key, [])
                if not len(ts_array) > 0:
                    if f"{outcome}_licks" in sess_grp:
                        del sess_grp[f"{outcome}_licks"]
                    sess_grp.create_dataset(f"{outcome}_licks", shape=(0, 0), dtype="float32")
                    print(f"{outcome}_licks...empty dataset written")
                    continue

                max_len = max(len(row[~pd.isnull(row)]) for row in ts_array)
                padded = np.full((len(ts_array), max_len), np.nan, dtype="float32")
                for t_idx, row in enumerate(ts_array):
                    clean_row = row[~pd.isnull(row)].astype("float32")
                    padded[t_idx, :len(clean_row)] = clean_row
                if f"{outcome}_licks" in sess_grp:
                    del sess_grp[f"{outcome}_licks"]
                sess_grp.create_dataset(f"{outcome}_licks", data=padded)
                print(f"{outcome}_licks...done")
    print("训练记录写入成功！")


def write_trial_daily_results_to_hdf5_from_cfg(cfg):
    """
    从 config 中构建路径并写入 trial_results 和 lick_times 到 HDF5 文件。
    - 每天一个 h5 文件
    - 每个文件内是 trial_0, trial_1, ...，每个 trial 下有 trial_results 和 lick_times
    """
    results_dir,h5_base_dir, trial_result_path, lick_paths = build_paths(cfg)

    # 读 trial_results
    trial_df = pd.read_csv(trial_result_path, header=None, index_col=0)
    trial_df.index = trial_df.index.astype(str)
    trial_df = trial_df.sort_index()

    # 读 lick data
    lick_data = {label: load_lick_sheets(path) for label, path in lick_paths.items()}

    # 遍历每个 session（一天）
    for session_key, row in trial_df.iterrows():
        date_key = session_key.split('_')[0]  # e.g. "20250729"
        
        # 输出路径：每天一个 h5
        h5_path = os.path.join(results_dir, f"{cfg['mouse_id']}_{date_key}.h5")

        # trial_results: 去掉 NaN
        trial_results = row.values.astype("float")
        last_valid_index = np.where(~np.isnan(trial_results))[0][-1]
        trial_results = trial_results[:last_valid_index]

        # 找到该 session 对应的 lick 数据 (合并4类lick)
        lick_times_all = []
        for outcome in ["Hit", "Miss", "CR", "FA"]:
            ts_array = lick_data[outcome].get(session_key, [])
            for lick_trial in ts_array:
                lick_trial_clean = lick_trial[~pd.isnull(lick_trial)].astype("float32")
                lick_times_all.append(lick_trial_clean)

        # 打开/新建 h5 文件，写入 trial 数据
        with h5py.File(h5_path, "a") as f:
            for i, res in enumerate(trial_results):
                trial_key = f"trial_{i}"
                if trial_key in f:
                    del f[trial_key]
                g = f.create_group(trial_key)

                g.create_dataset("trial_results", data=res)

                if i < len(lick_times_all):
                    g.create_dataset("lick_times", data=lick_times_all[i])
                else:
                    g.create_dataset("lick_times", shape=(0,), dtype="float32")

        print(f"✅ {session_key}: {len(trial_results)} trials saved to {h5_path}")

    print("🎉 所有 session 写入完成！")

