import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os
from datetime import datetime
from utils.preprocessing_behavior import bin_trials_and_compute_dprime
import seaborn as sns
import pandas as pd

def plot_metric_across_days(cfg, metric='dprime', save_dir=None):
    mouse_id = cfg['mouse_id']
    base_dir = cfg['base_dir']
    date_list = cfg['sessions']
    mouse_id_attr = mouse_id 

    # 拼接hdf5路径，假设文件名格式是 {mouse_id}_discrimination_data.h5，且都在results文件夹
    hdf5_path = os.path.join(base_dir, mouse_id, 'results', f"{mouse_id}_discrimination_data.h5")
    if save_dir is None:
        save_dir = os.path.join(base_dir, mouse_id, 'figures')
    os.makedirs(save_dir, exist_ok=True)

    assert metric in ['dprime', 'hit_fa', 'accuracy'], "metric must be one of: 'dprime', 'hit_fa', 'accuracy'"

    try:
        with h5py.File(hdf5_path, 'r') as f:
            training_history_0 = f['training_history'][()]
            mouse_id_attr = f.attrs.get('mouse_id', mouse_id)
        training_history = training_history_0.astype(str)
        history_dict = {row[0]: row[1] for row in training_history}
    except KeyError:
        print("[⚠] training_history 不存在，跳过历史标注")

    day_starts = []
    history_lines = []

    plt.figure(figsize=(36, 6))

    for day_index, date_str in enumerate(date_list):
        try:
            bins, summary = bin_trials_and_compute_dprime(hdf5_path, date_str)
            x = [day_index + i / len(bins) for i in range(len(bins))]

            if metric == 'dprime':
                y = [b['dprime'] for b in bins]
                color = 'teal'
                summary_value = summary['dprime']
                plt.plot(x, y, marker='o', linestyle='-', color=color, label='dprime', markersize=5)
                plt.hlines(summary_value, x[0], x[-1], colors=color, linestyles='-', alpha=0.5)
                if summary.get('dprime_200'):
                    summary_value_200 = summary['dprime_200']
                    plt.hlines(summary_value_200, x[0], x[-3], colors='darkslategrey', linestyles='-', alpha=0.5)
            elif metric == 'accuracy':
                y = [b['accuracy'] for b in bins]
                color = 'green'
                summary_value = summary["accuracy"]
                plt.plot(x, y, marker='o', linestyle='-', color=color, label='accuracy', markersize=5)
                plt.hlines(summary_value, x[0], x[-1], colors=color, linestyles='-', alpha=0.5)
                if summary.get("accuracy_200"):
                    summary_value_200 = summary["accuracy_200"]
                    plt.hlines(summary_value_200, x[0], x[-3], colors='darkgreen', linestyles='-', alpha=0.5)
            elif metric == 'hit_fa':
                hit_y = [b['hit_rate'] for b in bins]
                fa_y = [b['fa_rate'] for b in bins]
                plt.plot(x, hit_y, marker='o', linestyle='-', color='green', label='Hit Rate' if day_index == 0 else None)
                plt.plot(x, fa_y, marker='x', linestyle='-', color='red', label='FA Rate' if day_index == 0 else None)
                plt.hlines(summary['hit_rate'], x[0], x[-1], colors='green', linestyles='-', alpha=0.5)
                plt.hlines(summary['fa_rate'], x[0], x[-1], colors='red', linestyles='-', alpha=0.5)
                if summary.get('hit_rate_200'):
                    plt.hlines(summary['hit_rate_200'], x[0], x[-3], colors='darkgreen', linestyles='-', alpha=0.5)
                    plt.hlines(summary['fa_rate_200'], x[0], x[-3], colors='darkred', linestyles='-', alpha=0.5)
                y = hit_y
                color = None
            day_starts.append((x[0], y[0], date_str))

            date_formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
            if date_formatted in history_dict:
                history_lines.append((x[0], history_dict[date_formatted]))

        except Exception as e:
            print(f"[⚠️] 跳过 {date_str}: {e}")
            continue

    for x0, y0, label in day_starts:
        plt.scatter(x0, y0, color='blue', s=10, zorder=5)

    colors = ['orange', 'purple', 'green', 'blue', 'brown']
    for i, (x_line, label) in enumerate(history_lines):
        color = colors[i % len(colors)]
        plt.axvline(x_line, linestyle='--', color=color, alpha=0.7, label=label)

    plt.axhline(0, color='gray', linestyle='--')

    title_map = {
        'dprime': f"{mouse_id_attr} d' over cumulative training days",
        'accuracy': f"{mouse_id_attr} Accuracy over cumulative training days",
        'hit_fa': f"{mouse_id_attr} Hit Rate and FA Rate over training days"
    }
    ylabel_map = {
        'dprime': "d'",
        'accuracy': "Accuracy",
        'hit_fa': "Rate"
    }

    plt.title(title_map[metric], fontsize=14)
    plt.xlabel("Training Day", fontsize=12)
    plt.ylabel(ylabel_map[metric], fontsize=12)
    if metric == 'hit_fa':
        plt.ylim(0, 1.1)
    if metric == 'accuracy':
        plt.ylim(0.3, 1)
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.8), borderaxespad=0)
    plt.grid(alpha=0.3)
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    timestamp = datetime.now().strftime("%Y%m%d")
    save_path = os.path.join(save_dir, f"{mouse_id_attr}_{metric}_curve_{timestamp}.png")
    plt.savefig(save_path, dpi=300)
    print(f"[✔] 图像已保存至: {save_path}")
    plt.show()

def plot_dprime_across_days(hdf5_path, date_list, save_dir=None):
    with h5py.File(hdf5_path, 'r') as f:
        training_history_0 = f['training_history'][()]
        mouse_id = f.attrs['mouse_id']
    training_history = training_history_0.astype(str)
    history_dict = {row[0]: row[1] for row in training_history}

    day_starts = []
    history_lines = []

    plt.figure(figsize=(36, 6))

    for day_index, date_str in enumerate(date_list):
        try:
            bins = bin_trials_and_compute_dprime(hdf5_path, date_str)[0]
            y = [b['dprime'] for b in bins]
            x = [day_index + i / len(bins) for i in range(len(bins))]

            # 分开画：只连当天的线
            plt.plot(x, y, marker='o', linestyle='-', color='teal')

            day_starts.append((x[0], y[0], date_str))

            date_formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
            if date_formatted in history_dict:
                history_lines.append((x[0], history_dict[date_formatted]))

        except Exception as e:
            print(f"[⚠️] 跳过 {date_str}: {e}")
            continue

    # 每天起点标记
    for x0, y0, label in day_starts:
        plt.scatter(x0, y0, color='crimson', s=50, zorder=5)

    # 历史事件竖线
    colors = ['orange', 'purple', 'green', 'blue', 'brown']
    for i, (x_line, label) in enumerate(history_lines):
        color = colors[i % len(colors)]
        plt.axvline(x_line, linestyle='--', color=color, alpha=0.7, label=label)

    plt.axhline(0, color='gray', linestyle='--')
    plt.title(f"{mouse_id} d' over cumulative training days", fontsize=14)
    plt.xlabel("Training Day", fontsize=12)
    plt.ylabel("d'", fontsize=12)
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.8), borderaxespad=0)
    plt.grid(alpha=0.3)
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    if save_dir is None:
        save_dir = os.getcwd()
    timestamp = datetime.now().strftime("%Y%m%d")
    save_path = os.path.join(save_dir, f"{mouse_id}_dprime_curve_{timestamp}.png")
    plt.savefig(save_path, dpi=300)
    print(f"[✔] 图像已保存至: {save_path}")
    plt.show()

def plot_hit_fa_rate_across_days(hdf5_path, date_list, save_dir=None):
    with h5py.File(hdf5_path, 'r') as f:
        training_history_0 = f['training_history'][()]
        mouse_id = f.attrs['mouse_id']
    training_history = training_history_0.astype(str)
    history_dict = {row[0]: row[1] for row in training_history}

    day_starts = []
    history_lines = []

    plt.figure(figsize=(18, 6))

    for day_index, date_str in enumerate(date_list):
        try:
            bins = bin_trials_and_compute_dprime(hdf5_path, date_str)[0]

            x = [day_index + i / len(bins) for i in range(len(bins))]
            hit = [b['hit_rate'] for b in bins]
            fa = [b['fa_rate'] for b in bins]

            # 每天分别画 hit 和 fa 曲线
            plt.plot(x, hit, marker='o', linestyle='-', color='green')
            plt.plot(x, fa, marker='x', linestyle='-', color='red')

            # 记录起点
            day_starts.append((x[0], hit[0], fa[0], date_str))

            date_formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
            if date_formatted in history_dict:
                history_lines.append((x[0], history_dict[date_formatted]))

        except Exception as e:
            print(f"[⚠️] 跳过 {date_str}: {e}")
            continue

    # 每天起点高亮
    for x0, hit0, fa0, date_str in day_starts:
        plt.scatter(x0, hit0, color='darkgreen', s=50, zorder=5)
        plt.scatter(x0, fa0, marker='x', color='darkred', s=50, zorder=5)

    # 历史事件线
    colors = ['orange', 'purple', 'blue', 'brown']
    for i, (x_line, label) in enumerate(history_lines):
        color = colors[i % len(colors)]
        plt.axvline(x_line, linestyle='--', color=color, alpha=0.6, label=label)

    plt.title(f"{mouse_id} Hit Rate and FA Rate over Training Days", fontsize=14)
    plt.xlabel("Training Day", fontsize=12)
    plt.ylabel("Rate", fontsize=12)
    plt.ylim(0, 1.1)
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.8), borderaxespad=0)
    plt.grid(alpha=0.3)
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    if save_dir is None:
        save_dir = os.getcwd()
    timestamp = datetime.now().strftime("%Y%m%d")
    save_path = os.path.join(save_dir, f"{mouse_id} hit_fa_rates_curve_{timestamp}.png")
    plt.savefig(save_path, dpi=300)
    print(f"[✔] 图像已保存至: {save_path}")
    plt.show()

def plot_daily_dprime(hdf5_path, date_list, metric='dprime', save_dir=None):
    """
    为每一天绘制一个 d' 值的图像（每天一个点）。同时添加训练历史事件。
    metric: 'dprime', 'hit_rate', 'fa_rate' 之一
    """

    # 获取训练历史事件
    with h5py.File(hdf5_path, 'r') as f:
        training_history_0 = f['training_history'][()]
        mouse_id = f.attrs['mouse_id']
    training_history = training_history_0.astype(str)
    history_dict = {row[0]: row[1] for row in training_history}

    xs, ys = [], []
    history_lines = []

    for day_index, date_str in enumerate(date_list):
        try:
            summary = calculate_date_dprime(hdf5_path, date_str)
            xs.append(day_index)
            ys.append(summary[metric])

            # 若该天有历史事件，记录竖线位置信息
            date_formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
            if date_formatted in history_dict:
                history_lines.append((day_index, history_dict[date_formatted]))

        except Exception as e:
            print(f"[⚠️] 跳过 {date_str}: {e}")
            continue

    # === 绘图 ===
    plt.figure(figsize=(12, 5))
    plt.plot(xs, ys, marker='o', linestyle='-', color='royalblue', label="d'")

    # 添加训练历史事件的竖线
    colors = ['orange', 'purple', 'green', 'blue', 'brown']
    for i, (xline, label) in enumerate(history_lines):
        plt.axvline(xline, linestyle='--', color=colors[i % len(colors)], alpha=0.7, label=label)

    plt.title(f"{mouse_id} Daily {metric} across training days", fontsize=14)
    plt.xlabel("Training Day", fontsize=12)
    plt.ylabel("d'", fontsize=12)
    plt.axhline(0, color='gray', linestyle='--')
    #plt.legend()
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.8), borderaxespad=0)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # === 保存图像 ===
    if save_dir is None:
        save_dir = os.getcwd()
    timestamp = datetime.now().strftime("%Y%m%d")
    save_path = os.path.join(save_dir, f"{mouse_id}_daily_{metric}_{timestamp}.png")
    plt.savefig(save_path, dpi=300)
    print(f"[✔] 图像已保存至: {save_path}")
    plt.show()

def plot_daily_hit_fa_rate(hdf5_path, date_list, save_dir=None):
    """
    将 date_list 中每一天的整体 Hit Rate 和 FA Rate 画在一张图上。
    """
    with h5py.File(hdf5_path, 'r') as f:
        training_history_0 = f['training_history'][()]
        mouse_id = f.attrs['mouse_id']
    training_history = training_history_0.astype(str)
    history_dict = {row[0]: row[1] for row in training_history}

    xs, hit_ys, fa_ys = [], [], []
    history_lines = []

    for day_index, date_str in enumerate(date_list):
        try:
            summary = calculate_date_dprime(hdf5_path, date_str)
            xs.append(day_index)
            hit_ys.append(summary['hit_rate'])
            fa_ys.append(summary['fa_rate'])

            date_formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
            if date_formatted in history_dict:
                history_lines.append((day_index, history_dict[date_formatted]))

        except Exception as e:
            print(f"[⚠️] 跳过 {date_str}: {e}")
            continue

    # === 绘图 ===
    plt.figure(figsize=(12, 6))
    plt.plot(xs, hit_ys, marker='o', linestyle='-', color='green', label='Hit Rate')
    plt.plot(xs, fa_ys, marker='x', linestyle='-', color='red', label='FA Rate')

    # 添加训练历史事件竖线
    colors = ['orange', 'purple', 'blue', 'brown']
    for i, (xline, label) in enumerate(history_lines):
        color = colors[i % len(colors)]
        plt.axvline(xline, linestyle='--', color=color, alpha=0.6, label=label)

    plt.title(f"{mouse_id} Daily Hit/FA Rate across training days", fontsize=14)
    plt.xlabel("Training Day", fontsize=12)
    plt.ylabel("Rate", fontsize=12)
    plt.ylim(0, 1.1)
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.8), borderaxespad=0)
    plt.grid(alpha=0.3)
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    # === 保存图像 ===
    if save_dir is None:
        save_dir = os.getcwd()
    timestamp = datetime.now().strftime("%Y%m%d")
    save_path = os.path.join(save_dir, f"{mouse_id}_daily_hit_fa_rate_{timestamp}.png")
    plt.savefig(save_path, dpi=300)
    print(f"[✔] 图像已保存至: {save_path}")

    plt.show()

def interpolate_colors(start_color, end_color, n):
    return np.linspace(start_color, end_color, n)

def plot_dprime_bins(hdf5_path, date_list, metric='dprime',save_dir=None):
    """
    绘制按 bin 计算的 d' / hit_rate / fa_rate 曲线
    - date_list: str 或 list[str]
    - metric: 'dprime', 'hit_rate', 'fa_rate' 之一
    """
    if isinstance(date_list, str):
        date_list = [date_list]
    
    with h5py.File(hdf5_path, 'r') as f:
        mouse_id = f.attrs['mouse_id']

    n_dates = len(date_list)
    start_color = np.array([1, 0.8, 1])  # 浅品红
    end_color = np.array([1, 0, 1])      # 深品红
    colors = interpolate_colors(start_color, end_color, n_dates)
    line_widths = 3

    max_bin_len = 0
    all_curves = []

    for date in date_list:
        dprimes, _ = bin_trials_and_compute_dprime(hdf5_path, date)
        y_vals = [dp[metric] for dp in dprimes]
        all_curves.append(y_vals)
        max_bin_len = max(max_bin_len, len(y_vals))

    fig, ax = plt.subplots(figsize=(8, 5))

    if len(date_list) < 2:
        x = np.arange(1, len(all_curves[0]) + 1)
        ax.plot(
            x, all_curves[0],
            label=date_list[0],
            color='#ADD8E6',  # 浅蓝色
            linewidth=3,
            marker='o',
            markersize=6
        )
    else:
        for i, (date, y_vals) in enumerate(zip(date_list, all_curves)):
            x = np.arange(1, len(y_vals) + 1)
            ax.plot(
                x, y_vals,
                label=date,
                color=colors[i],  # 渐变的品红色
                linewidth=3,
                marker='o',
                markersize=6
            )
    
    # 添加网格线
    ax.grid(True, linestyle='--', alpha=0.6)
    
    # 图例和布局
    ax.set_xlabel("Bin")
    ax.set_ylabel(metric)
    ax.set_title(f"{mouse_id} {metric} across bins")
    ax.set_xticks(np.arange(1, max_bin_len + 1))  # 1 到最大的 bin 编号
    ax.set_xticklabels([str(i) for i in range(1, max_bin_len + 1)])  # 强制显示为整数标签
    plt.legend(loc='center left', bbox_to_anchor=(1.02, 0.8), borderaxespad=0)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    if save_dir is None:
        save_dir = os.getcwd()
    timestamp = datetime.now().strftime("%Y%m%d")
    save_path = os.path.join(save_dir, f"{mouse_id}_daily_hit_fa_rate_{timestamp}.png")
    plt.savefig(save_path, dpi=300)

    plt.show()

##Special events
def plot_lick_histograms_by_case(hdf5_path, case, trial_setting, date_list, save_dir=None):
    '''
    trial_setting = [0, 1.25, 2.25, 4.25, 5.25, 9.75]
    '''
    case_key = f"{case}_licks"
    count_key = "trial_results"
    case_dic = {'Hit': 1, 'Miss': 2, 'FA': 3, 'CR': 4}

    with h5py.File(hdf5_path, 'r') as f:
        mouse_id = f.attrs['mouse_id']
        root = f['training_records']
        time_dict = {}
        trial_counts = []

        for date in date_list:
            if date not in root:
                continue
            date_group = root[date]
            date_timesteps = []
            total_case_trials = 0

            for session_name in date_group:
                session_group = date_group[session_name]
                if case_key in session_group:
                    licks = session_group[case_key][()]  # shape: (n_trials, n_licks)
                    licks = licks[:, 1:]  # 去掉第一列索引或占位
                    trial_result = session_group[count_key][()]

                    flattened = licks.flatten()
                    times = flattened[~np.isnan(flattened)]
                    date_timesteps.append(times)

                    total_case_trials += np.count_nonzero(trial_result == case_dic[case])

            if date_timesteps:
                time_dict[date] = np.concatenate(date_timesteps)
                trial_counts.append(total_case_trials)
            else:
                time_dict[date] = np.array([])
                trial_counts.append(0)

    if not time_dict:
        print("未找到任何有效的数据。")
        return

    # 绘图部分
    fig, axes = plt.subplots(len(time_dict), 1, figsize=(20, 4 * len(time_dict)))
    if len(time_dict) == 1:
        axes = [axes]

    bin_width = 0.1
    bins = np.arange(0, 9 + bin_width, bin_width)

    for i, (label, values) in enumerate(time_dict.items()):
        ax = axes[i]
        ax.hist(values, bins=bins, edgecolor='black', align='left', color='lightblue')
        ax.axvspan(trial_setting[1], trial_setting[2], color='grey', alpha=0.5)
        ax.axvspan(trial_setting[2], trial_setting[3], color='orange', alpha=0.5)

        ax.set_xlim([0, 6])
        ax.set_ylim([0, 100])
        ax.grid(True)

        #ax.set_title(f"{label} — {case} trials: {trial_counts[i]}", fontsize=20)
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(50))

        ax.tick_params(axis='x', labelsize=22, colors='black')
        ax.tick_params(axis='y', labelsize=22, colors='black')

    plt.tight_layout()
    if save_dir is None:
        save_dir = os.getcwd()
    timestamp = datetime.now().strftime("%Y%m%d")
    save_path = os.path.join(save_dir, f"{mouse_id}_lick_histograms_{case}_{timestamp}.png")
    plt.savefig(save_path, dpi=300)
    plt.show()

    print(f"图像已保存到: {save_path}")

def plot_lick_histograms_by_day(hdf5_path, trial_setting, date_list, save_dir=None):
    """
    每天画一张图，每张图包含 Hit / Miss / FA / CR 四个 licking 时间的直方图。

    参数:
    - hdf5_path: HDF5 文件路径
    - trial_setting: 例如 [0, 1.25, 2.25, 4.25, 5.25, 9.75]
    - date_list: 例如 ['20250407']
    - save_dir: 保存图像路径
    """
    case_dic = {'Hit': 1, 'Miss': 2, 'FA': 3, 'CR': 4}

    with h5py.File(hdf5_path, 'r') as f:
        mouse_id = f.attrs['mouse_id']
        root = f['training_records']
        all_days_licks = {}

        for date in date_list:
            if date not in root:
                print(f"日期 {date} 不在数据中，跳过。")
                continue

            date_group = root[date]
            lick_times_by_case = {case: [] for case in case_dic.keys()}

            for session_name in date_group:
                session_group = date_group[session_name]

                for case in case_dic.keys():
                    case_key = f"{case}_licks"
                    if case_key not in session_group:
                        continue

                    licks = session_group[case_key][()]
                    licks = licks[:, 1:]  # 去除 trial index 列

                    # 展平并去除 NaN
                    flattened = licks.flatten()
                    valid_times = flattened[~np.isnan(flattened)]

                    lick_times_by_case[case].append(valid_times)

            # 整合所有 session 的该 case licking 数据
            daily_lick_times = {}
            for case in case_dic.keys():
                if lick_times_by_case[case]:
                    daily_lick_times[case] = np.concatenate(lick_times_by_case[case])
                else:
                    daily_lick_times[case] = np.array([])

            all_days_licks[date] = daily_lick_times

    if not all_days_licks:
        print("未找到任何有效的数据。")
        return

    # 逐天绘图
    for date, lick_dict in all_days_licks.items():
        fig, axes = plt.subplots(4, 1, figsize=(8, 9), sharex=True)
        bin_width = 0.1
        bins = np.arange(0, 9 + bin_width, bin_width)

        for i, case in enumerate(['Hit', 'Miss', 'FA', 'CR']):
            ax = axes[i]
            values = lick_dict[case]

            ax.hist(values, bins=bins, edgecolor='black', align='left', color='lightblue')
            ax.axvspan(trial_setting[1], trial_setting[2], color='grey', alpha=0.5)
            ax.axvspan(trial_setting[2], trial_setting[3], color='orange', alpha=0.5)

            ax.set_xlim([0, 6])
            ax.set_ylim([0, 150])
            ax.grid(True)

            #ax.set_title(f"{date} — {case}", fontsize=20)
            ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
            ax.yaxis.set_major_locator(ticker.MultipleLocator(50))
            ax.tick_params(axis='x', labelsize=20, colors='black')
            ax.tick_params(axis='y', labelsize=20, colors='black')

        axes[-1].set_xlabel("Lick time (s)", fontsize=24)
        plt.tight_layout()

        if save_dir is None:
            save_dir = os.getcwd()
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d")
        save_path = os.path.join(save_dir, f"{mouse_id}_lick_histograms_all_cases_{date}_{timestamp}.png")
        plt.savefig(save_path, dpi=300)
        plt.show()

        print(f"图像已保存到: {save_path}")

def plot_lick_raster(hdf5_path, case, date_list, save_dir=None):
    """
    以 raster plot 形式绘制舔水事件，每个日期一个子图。
    每一行代表一个 trial，小柱子代表该 trial 的舔水 timestamp。

    参数：
        hdf5_path (str): HDF5 文件路径
        case (str): 选择的 trial 类型 ('Hit', 'FA', 'CR', 'Miss')
        date_list (list[str]): 指定绘图的训练日期列表，例如 ['20250220', '20250221']
    """
    case_key = f"{case}_licks"
    count_key = "trial_results"
    case_dic = {'Hit': 1, 'Miss': 2, 'FA': 3, 'CR': 4}

    all_date_data = {}  # {date: [(trial_label, [timestamps])]}
    
    with h5py.File(hdf5_path, 'r') as f:
        mouse_id = f.attrs['mouse_id']
        root = f['training_records']
        
        for date in date_list:
            if date not in root:
                continue
            date_group = root[date]
            date_data = []
            trial_label = 0  # 从0开始标记当天的 trial label
            trial_counts = []

            for session_name in date_group:
                session_group = date_group[session_name]
                if case_key in session_group:
                    licks = session_group[case_key][()]  # shape: (n_trials, n_licks + 1)
                    licks = licks[:, 1:]  # 去掉 trial label 列
                    trial_result = session_group[count_key][()]

                    for trial_idx, row in enumerate(licks):
                        #if trial_result[trial_idx] != case_dic[case]:
                            #continue
                        timestamps = row[~np.isnan(row)]
                        #if len(timestamps) == 0: 
                            #continue
                        date_data.append((trial_label, timestamps))
                        trial_label += 1

            if date_data:
                all_date_data[date] = date_data

    if not all_date_data:
        print("未找到任何有效的数据。")
        return

    num_dates = len(all_date_data)
    if case_dic[case]==1:
        fig, axes = plt.subplots(num_dates, 1, figsize=(20, 15 * num_dates), sharex=True)
    elif case_dic[case]==3:
        fig, axes = plt.subplots(num_dates, 1, figsize=(20, 8* num_dates), sharex=True)
    else:
        fig, axes = plt.subplots(num_dates, 1, figsize=(20, 6 * num_dates), sharex=True)
    if num_dates == 1:
        axes = [axes]

    for ax, (date, trials) in zip(axes, all_date_data.items()):
        for trial_label, timestamps in trials:
            ax.hlines(trial_label, 0, 11, color='lightgray', linewidth=0.8)
            ax.vlines(timestamps, trial_label - 0.4, trial_label + 0.4, color='blue')

        ax.axvspan(1.25, 2.75, color='grey', alpha=0.5) 
        ax.axvspan(2.75, 4.75, color='orange', alpha=0.5)
        
        #ax.set_ylabel(f"{date}\nTrial", fontsize='medium', rotation='horizontal', labelpad=50)
        plt.xlim(0, 11)
        ax.set_ylim(-1, len(trials))
        ax.grid(True, axis='x', linestyle='--', alpha=0.5)
        #ax.set_title(f"{date} - {case}", fontsize='35')
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(20))

    # 设置 x、y 轴刻度字体大小为25，颜色为黑色
        ax.tick_params(axis='x', labelsize=22, colors='black')
        ax.tick_params(axis='y', labelsize=22, colors='black')

    #axes[-1].set_xlabel("Time (s)", fontsize='large')
    
    plt.tight_layout()
    if save_dir is None:
        save_dir = os.getcwd()
    timestamp = datetime.now().strftime("%Y%m%d")
    save_path = os.path.join(save_dir, f"{mouse_id}_lick_raster_{case}_{timestamp}.png")
    plt.savefig(save_path, dpi=300)
    plt.show()


def plot_go_nogo_lick_raster_by_day(hdf5_path, trial_setting, date_list, save_dir=None):
    """
    绘制 Go (Hit + Miss) 和 No-Go (FA + CR) trial 的舔水 raster 图。
    每天两个子图（Go 和 No-Go），每个点代表一个舔水事件，颜色区分 trial 类型。

    trial_setting: [0, 1.25, 2.25, 4.25, 5.25, 9.75]
    """
    case_color = {1: 'green', 2: 'lightcoral', 3: 'red', 4: 'lightgreen'}
    case_color_2 = {1: 'lightgray', 2: 'lightcoral', 3: 'lightgray', 4: 'lightgreen'}
    case_keys = {'Go': ['Hit', 'Miss'], 'NoGo': ['FA', 'CR']}
    case_labels = {'Hit': 1, 'Miss': 2, 'FA': 3, 'CR': 4}

    def pad_to_same_width(arrays):
        max_cols = max(a.shape[1] for a in arrays)
        padded = [np.pad(a, ((0, 0), (0, max_cols - a.shape[1])), constant_values=np.nan) for a in arrays]
        return padded

    with h5py.File(hdf5_path, 'r') as f:
        mouse_id = f.attrs['mouse_id']
        root = f['training_records']

        for date in date_list:
            if date not in root:
                continue

            date_group = root[date]
            go_trials, nogo_trials = [], []
            go_breaks, nogo_breaks = [], []  # session 分界线
            go_index, nogo_index = 0, 0

            for session_name in sorted(date_group.keys()):
                session_group = date_group[session_name]

                go_session_trials = []
                nogo_session_trials = []

                for case in ['Hit', 'Miss', 'FA', 'CR']:
                    case_key = f"{case}_licks"
                    if case_key not in session_group:
                        continue

                    raw = session_group[case_key][()]  # (n_trials, n_timestamps+1)
                    trial_indices = raw[:, [0]]
                    timestamps = raw[:, 1:]
                    n_trials = raw.shape[0]

                    labels = np.full((n_trials, 1), case_labels[case])
                    data = np.hstack([labels, trial_indices, timestamps])

                    if case in case_keys['Go']:
                        go_session_trials.append(data)
                    else:
                        nogo_session_trials.append(data)

                # 合并 Go trial，并按第二列（trial index）排序
                if go_session_trials:
                    go_combined = np.vstack(pad_to_same_width(go_session_trials))
                    go_combined = go_combined[np.argsort(go_combined[:, 1])]
                    go_trials.append(go_combined)
                    go_breaks.append(go_combined.shape[0])
                if nogo_session_trials:
                    nogo_combined = np.vstack(pad_to_same_width(nogo_session_trials))
                    nogo_combined = nogo_combined[np.argsort(nogo_combined[:, 1])]
                    nogo_trials.append(nogo_combined)
                    nogo_breaks.append(nogo_combined.shape[0])

            # 合并所有 session 的 trial，并重编号
            def process_trials(trial_list, breaks):
                if not trial_list:
                    return None, []
                full = np.vstack(pad_to_same_width(trial_list))
                trial_rows = []
                current_trial = 0
                for row in full:
                    trial_type = int(row[0])
                    timestamps = row[2:][~np.isnan(row[2:])]
                    trial_rows.append((trial_type, current_trial, timestamps))
                    current_trial += 1
                # 计算 session 分隔线位置
                session_lines = np.cumsum(breaks)[:-1]
                return trial_rows, session_lines

            go_data, go_lines = process_trials(go_trials, go_breaks)
            nogo_data, nogo_lines = process_trials(nogo_trials, nogo_breaks)

            # 绘图
            fig, axes = plt.subplots(2, 1, figsize=(20,30), sharex=True)
            for idx, (label, data, lines) in enumerate([("Go Trials", go_data, go_lines), ("No-Go Trials", nogo_data, nogo_lines)]):
                ax = axes[idx]
                if data:
                    for trial_type, trial_idx, timestamps in data:
                        ax.hlines(trial_idx, 0, 10, color=case_color_2[trial_type], linewidth=0.8,zorder=0)  # 添加灰色 trial 行
                        ax.vlines(timestamps, trial_idx - 0.6, trial_idx + 0.6, color=case_color[trial_type], linewidth=1,zorder=1)

                    for line_y in lines:
                        ax.axhline(line_y - 0.5, color='blue', linestyle='--', linewidth=1.5)
                ax.axvspan(trial_setting[1], trial_setting[2], color='grey', alpha=0.3)
                ax.axvspan(trial_setting[2], trial_setting[3], color='orange', alpha=0.3)
                ax.set_ylim(-1, len(data) if data else 1)
                ax.set_xlim(0, 10)
                ax.set_title(f"{date} - {label}", fontsize=24)
                ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
                ax.yaxis.set_major_locator(ticker.MultipleLocator(50))
                ax.tick_params(axis='x', labelsize=18)
                ax.tick_params(axis='y', labelsize=18)
                ax.grid(True, axis='x', linestyle='--', alpha=0.5)

            plt.tight_layout()
            if save_dir is None:
                save_dir = os.getcwd()
            timestamp = datetime.now().strftime('%Y%m%d')
            save_path = os.path.join(save_dir, f"{mouse_id}_raster_{date}_{timestamp}.png")
            plt.savefig(save_path, dpi=300)
            plt.show()
            print(f"{date} 图像已保存至: {save_path}")


def plot_outcome_heatmap(hdf5_path, date_list):
    """
    从 HDF5 文件中读取指定日期列表中所有 session 的 trial_results，并绘制热图。

    Parameters:
    - hdf5_path: str, HDF5 文件路径
    - date_list: list of str, 例如 ['20250224', '20250225', '20250421']
    """
    outcome_matrix = []
    y_ticks = []

    with h5py.File(hdf5_path, 'r') as f:
        training_records = f['training_records']

        for date in date_list:
            if date not in training_records:
                print(f"[跳过] 日期 {date} 不在 HDF5 中")
                continue

            date_group = training_records[date]
            session_keys = list(date_group.keys())

            if not session_keys:
                print(f"[跳过] 日期 {date} 下无 session 数据")
                continue

            for session in session_keys:
                session_group = date_group[session]
                if 'trial_results' not in session_group:
                    print(f"[跳过] 日期 {date} session {session} 中无 trial_results")
                    continue

                results = session_group['trial_results'][:]
                outcome_matrix.append(results)
                y_ticks.append(f"{date} ({session})")  # 标注 session

    if not outcome_matrix:
        print("没有可用数据绘图")
        return

    # 构造 DataFrame（填 NaN 再填 0）
    outcomes_df = pd.DataFrame(outcome_matrix).fillna(0).astype(int)

    # 绘图
    plt.figure(figsize=(15, len(outcomes_df)*0.5 + 2))
    cmap = sns.color_palette('Accent', n_colors=5)

    heatmap = sns.heatmap(
        outcomes_df,
        cmap=cmap,
        cbar=False,
        xticklabels=10,
        yticklabels=y_ticks
    )

    # 添加颜色条
    colorbar = heatmap.figure.colorbar(
        heatmap.collections[0],
        ticks=[0, 1, 2, 3, 4],
        orientation='vertical'
    )
    colorbar.set_ticklabels(["NA", "Hit", "Miss", "FA", "CR"])
    plt.title("Trial Outcomes Heatmap", fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_cut_and_full_session(hdf5_path,datelist):
    '''这个函数包含两张图：
    1.柱状图：把200和300放在一起作比较
    2.直方图：dprime_300-dprime_200的分布'''
    dprime_all_list = []
    dprime_200_list = []
    delta_list = []
    valid_dates = []

    # 从 HDF5 中读取 mouse_id
    with h5py.File(hdf5_path, 'r') as f:
        mouse_id = f.attrs['mouse_id']

    # 遍历日期并提取 summary
    for date_str in datelist:
        try:
            _, summary = bin_trials_and_compute_dprime(hdf5_path, date_str)
            #print(summary)
            d_all = summary.get('dprime', 0)
            d_200 = summary.get('dprime_200', None)

            dprime_all_list.append(d_all)

            if d_200 is not None:
                dprime_200_list.append(d_200)
                delta_list.append(d_all - d_200)
            else:
                dprime_200_list.append(0)  # 仍显示柱子
                print(f"[ℹ️] {date_str} 无 dprime_200，仅显示 d' all，不计入 delta")

            valid_dates.append(date_str)
        except Exception as e:
            print(f"[⚠️] 跳过 {date_str}: {e}")

    # 图1：柱状图
    x = np.arange(len(valid_dates))
    width = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar(x - width / 2, dprime_all_list, width, label="d' all", color='teal')
    plt.bar(x + width / 2, dprime_200_list, width, label="d' 200", color='darkslategrey')
    plt.xticks(x, valid_dates, rotation=45)
    plt.ylabel("d'")
    plt.title(f"{mouse_id} - Daily d' (all vs. first 200 trials)")
    plt.legend()
    plt.tight_layout()

    #if save_dir is None:
    #    save_dir = os.getcwd()
    timestamp = datetime.now().strftime("%Y%m%d")
    #save_path1 = os.path.join(save_dir, f"{mouse_id}_dprime_bar_{timestamp}.png")
    #plt.savefig(save_path1, dpi=300)
    #print(f"[✔] 图1已保存至: {save_path1}")
    plt.show()

    # 图2：差值直方图
    #print(delta_list)
    plt.figure(figsize=(8, 5))
    plt.hist(delta_list, bins=np.arange(min(delta_list), max(delta_list) + 0.05, 0.05), color='slateblue', edgecolor='black')
    plt.xlabel("d' all - d' 200")
    plt.ylabel("Frequency")
    plt.title(f"{mouse_id} - Histogram of d' all minus d' 200")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    #save_path2 = os.path.join(save_dir, f"{mouse_id}_dprime_diff_hist_{timestamp}.png")
    #plt.savefig(save_path2, dpi=300)
    #print(f"[✔] 图2已保存至: {save_path2}")
    plt.show()

    #图3：dprime_all直方图
    plt.figure(figsize=(8, 5))
    plt.hist(dprime_all_list, bins=np.arange(min(dprime_all_list), max(dprime_all_list) + 0.05, 0.05), color='teal', edgecolor='black')
    plt.xlabel("d' all")
    plt.ylabel("Frequency")
    plt.title(f"{mouse_id} - Histogram of d' all")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    #save_path2 = os.path.join(save_dir, f"{mouse_id}_dprime_diff_hist_{timestamp}.png")
    #plt.savefig(save_path2, dpi=300)
    #print(f"[✔] 图2已保存至: {save_path2}")
    plt.show()

    #图4：dprime_200直方图
    plt.figure(figsize=(8, 5))
    plt.hist(dprime_200_list, bins=np.arange(min(dprime_200_list), max(dprime_200_list) + 0.05, 0.05), color='darkslategrey', edgecolor='black')
    plt.xlabel("d' all")
    plt.ylabel("Frequency")
    plt.title(f"{mouse_id} - Histogram of d'200")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    #save_path2 = os.path.join(save_dir, f"{mouse_id}_dprime_diff_hist_{timestamp}.png")
    #plt.savefig(save_path2, dpi=300)
    #print(f"[✔] 图2已保存至: {save_path2}")
    plt.show()

def plot_hit_fa_scatter(hdf5_path, date_list, save_dir=None):
    with h5py.File(hdf5_path, 'r') as f:
        training_history_0 = f['training_history'][()]
        mouse_id = f.attrs['mouse_id']
    training_history = training_history_0.astype(str)
    history_dict = {row[0]: row[1] for row in training_history}

    day_starts = []
    history_lines = []

    plt.figure(figsize=(72, 3))

    # 定义颜色和标签
    color_map = {
        4.0: ('olive', 'Hit'),
        3.0: ('pink', 'Miss'),
        2.0: ('magenta', 'FA'),
        1.0: ('lightgreen', 'CR'),
    }
    
    for day_index, date_str in enumerate(date_list):
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
                trial_result = date_group[sess_key]["trial_results"][:]*(-1)+5
                all_trials.append(trial_result)

        trial_seq = np.concatenate(all_trials)
        print(len(trial_seq))

        try:
            x = [day_index * 300 + i for i in range(len(trial_seq))]

            for val, (color, label) in color_map.items():
                indices = np.where(trial_seq == val)[0]
                if len(indices) == 0:
                    continue
                plt.scatter(np.array(x)[indices], trial_seq[indices], color=color, label=label, s=10)

            # 起点记录
            day_starts.append((x[0], trial_seq[0], date_str))

            date_formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:]}"
            if date_formatted in history_dict:
                history_lines.append((x[0], history_dict[date_formatted]))

        except Exception as e:
            print(f"[⚠️] 跳过 {date_str}: {e}")
            continue

    # 每天起点高亮
    for x0, trial_seq0, date_str in day_starts:
        plt.scatter(x0, trial_seq0, color='blue', s=50, zorder=5)

    # 历史事件线
    line_colors = ['orange', 'purple', 'blue', 'brown']
    for i, (x_line, label) in enumerate(history_lines):
        color = line_colors[i % len(line_colors)]
        plt.axvline(x_line, linestyle='--', color=color, alpha=0.6, label=label)

    plt.title(f"{mouse_id} trial results over days", fontsize=14)
    plt.xlabel("Training Trials", fontsize=12)
    plt.ylabel("Result", fontsize=12)
    plt.ylim(0, 5)

    # 🔥 去重图例
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(1.02, 0.8), borderaxespad=0)

    plt.grid(alpha=0.4)
    plt.tight_layout(rect=[0, 0, 0.85, 1])

    if save_dir is None:
        save_dir = os.getcwd()
    timestamp = datetime.now().strftime("%Y%m%d")
    save_path = os.path.join(save_dir, f"{mouse_id} training_results_{timestamp}.png")
    plt.savefig(save_path, dpi=300)
    print(f"[✔] 图像已保存至: {save_path}")
    plt.show()