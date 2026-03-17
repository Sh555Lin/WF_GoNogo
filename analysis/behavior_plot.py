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
from utils.wf_utils import *


def plot_individual_metrics_from_csv(mouseid, csv_path, save_dir):
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
        
        ax.set_title(f'{mouseid} {title} Across Training States', 
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
            save_path = os.path.join(save_dir, f'{metric}_of_{mouseid}_by_session.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.savefig(f'{metric}_of_{mouseid}_by_session.png', dpi=300, bbox_inches='tight')
            print(f"Plot saved to {metric}_by_session.png")
        
        plt.show()

base_dir = "/Volumes/Data_attention/Transfer learning/LinShu/DATA_linshu/000 Widefield"
mouse_id = "A095"
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
plot_individual_metrics_from_csv(mouseid=mouse_id,csv_path=csv_path,save_dir=save_dir)