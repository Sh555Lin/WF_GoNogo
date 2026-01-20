#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 09:19:56 2025

@author: yatangli
"""

from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import os
from glob import glob
import pandas as pd
import sys
from PIL import Image, ImageDraw, ImageFont
from tifffile import imwrite,imread
import pathlib

abspath = os.path.abspath(__file__)
current_dir = os.path.dirname(abspath)
parent_dir = os.path.dirname(current_dir)
os.chdir(parent_dir)
if parent_dir not in sys.path: 
    sys.path.append(parent_dir)
from utils.wf_utils import *


#%%
def pre_process_wf(rawPath):
# config_path = "/home/lsh/WF_GoNogo/config/A095_config.yaml"
# rawPath = os.path.join(base_dir, mouse_id, session)
    if not os.path.exists(rawPath):
        print(f"The specified path does not exist: {rawPath}")
        
    
    # mask save in ./rawPath/QualityControl/brain_mask.npy
    mask_path = os.path.join(rawPath, "QualityControl/brain_mask.npy")
    if os.path.exists(mask_path):
        mask = np.load(mask_path)
        # print(f"Mask shape: {mask.shape}")
        show_one_image(mask, title="Brain Mask")
    else:
        print(f"The specified mask path does not exist: {mask_path}")
        mask = None

    
    # n_preview = 300 # set to None to process all images, 
    # or set to an integer to limit the number of images processed for preview
    
    # if mask is not None, apply the mask to the images
    stack, files_470 = images2tiff_two_channel(rawPath,
        mask=mask)
    print("The shape of the processed images:", stack.shape)
    
    # visualize the first image of each channel
    images = list(stack[0])
    show_images(images, titles=['470 channel', '405 channel'])
    
    
    stack_mean = np.mean(stack, axis=(2, 3))
    print("The shape of the stack_mean:", stack_mean.shape)
    outlier_index_470, outlier_index_405 = detect_lum_outlier(stack_mean, 
        threshold=6, plot=True)
    
    stack_corrected = correct_lum_outlier(stack, outlier_index_470,
        outlier_index_405, plot=False)
    
    del stack
    #% calculate delta_f/f for 470 channel
    # the 'delta_ff_3d' is time consuming, optimize 'cal_base' may help
    
    
    # stack_470_dff = delta_ff_3d(stack_corrected[:, 0, :, :], win=150)
    # # print("The shape of stack_470_dff:", stack_470_dff.shape)
    
    # stack_405_dff = delta_ff_3d(stack_corrected[:, 1, :, :], win=150)
    # # print("The shape of stack_405_dff:", stack_405_dff.shape)
    
    
    
    
    dff = delta_ff_3d_2ch(stack_corrected[:, 0, :, :],stack_corrected[:, 1, :, :], win=150)
    
    del stack_corrected
    #%
    
    # dff = stack_470_dff - stack_405_dff
    # Scale the dff to range of 16 bit
    dff_min = dff.min()
    dff_max = dff.max()
    range_val = dff_max - dff_min
    
    dff_uint16 = np.empty(dff.shape, dtype=np.uint16)
    
    for i in tqdm(range(dff.shape[0])):  # 遍历 29,927 帧
        normalized_frame = 65535 * (dff[i] - dff_min) / range_val
        dff_uint16[i] = normalized_frame.astype(np.uint16)
        
    dff = dff_uint16
    del dff_uint16
    
    # dff = 65535 * (dff - dff_min) / (dff_max - dff_min)
    
    #%
    
    
    
    timePath = os.path.join(rawPath, 'time')
    trial_path = os.path.join(timePath, 'trial_timestamp.npy')
    # if not os.path.exists(trial_path):
    #     config_path = os.path.join(os.path.abspath('.'),'config/A095_config.yaml')
    #     preprocess_mice(config_path)
    trial_timestamp = np.load(trial_path)
    stim_path = os.path.join(timePath, 'stim_timestamp.npy')
    stim_timestamp = np.load(stim_path)
    wf_path = os.path.join(timePath, 'widefield_timestamps_blue.npy')
    wf_timestamp = np.load(wf_path)
    stim_path = os.path.join(timePath, 'stim_timestamp.npy')
    stim_timestamp = np.load(stim_path)
    wf_path = os.path.join(timePath, 'widefield_timestamps_blue.npy')
    wf_timestamp = np.load(wf_path)
    wf_timestamp = wf_timestamp/1e3
    print(wf_timestamp.shape)
    plt.plot(trial_timestamp,'.')
    plt.show()
    plt.plot(stim_timestamp,'r.')
    plt.show()
    plt.plot(wf_timestamp,'.')
    plt.show()
    
    wf_trial_0_idx,wf_trial_0_diff = find_closest_indices(wf_timestamp, trial_timestamp[:,0])
    wf_trial_1_idx,wf_trial_1_diff = find_closest_indices(wf_timestamp, trial_timestamp[:,1])
    plt.plot(wf_trial_0_idx[1:]-wf_trial_0_idx[:-1],'.')
    plt.plot(wf_trial_1_idx[1:]-wf_trial_1_idx[:-1],'.')
    plt.show()
    wf_sf = 10 #Hz
    trial_duration = (wf_trial_1_idx-wf_trial_0_idx)/wf_sf
    _idx = np.logical_or(trial_duration==5.2,trial_duration==9.7)
    wf_trial_1_idx[_idx] = wf_trial_1_idx[_idx]+1
    plt.plot((wf_trial_1_idx-wf_trial_0_idx)/wf_sf,'.')
    plt.show()
    plt.plot(wf_timestamp[18001:wf_trial_1_idx[-1]+6]-wf_timestamp[18000:wf_trial_1_idx[-1]+5],'.')
    plt.show()
    # acutal trial duration is 5.25 s or  9.75 s
    
    
    dff_text = np.copy(dff)
    for trial,i in enumerate(wf_trial_0_idx):
        if i <dff_text.shape[0]:
            dff_text[i,:,:] = add_text_to_image(dff[i,:,:], trial+1, status="on")
    for trial,i in enumerate(wf_trial_1_idx):
        if i <dff_text.shape[0]:
            dff_text[i,:,:] = add_text_to_image(dff[i,:,:], trial+1, status="off")
    
    
    print('Text added.')
    # save the dff_scaled as tiff file
    save_path = os.path.join(rawPath, f'{mouse_id}_{session}_dff.tif')
    imwrite(save_path, dff_text.astype(np.uint16), imagej=True)
    print('The dff tiff file saved to:', save_path)
    
    del dff_text
    #%
    # same averaged images: across all, Hit, CR, FA, Miss trials
    trial_type_path = os.path.join(timePath, 'trial_type.npy')
    if not os.path.exists(trial_type_path):
        trial_type = load_trial_type(rawPath, timePath)
    else:    
        trial_type = np.load(trial_type_path)
    plt.plot(trial_type,'.')
    
    dff_mean_all = compute_trial_mean(dff,wf_trial_0_idx,wf_trial_1_idx)
    save_path = os.path.join(rawPath, f'{mouse_id}_{session}_dff_mean_all.tif')
    imwrite(save_path, dff_mean_all.astype(np.uint16), imagej=True)
    print('The dff tiff file saved to:', save_path)
    
    
    type_list = ['Hit','Miss', 'FA','CR']
    for i in range(1,5):
        dff_mean = compute_trial_mean(dff,wf_trial_0_idx[trial_type==i],wf_trial_1_idx[trial_type==i])
        save_path = os.path.join(rawPath, f'{mouse_id}_{session}_dff_mean_{type_list[i-1]}.tif')
        imwrite(save_path, dff_mean.astype(np.uint16), imagej=True)
        print('The dff tiff file saved to:', save_path)



#%%
# base_dir = "/Volumes/Data_attention/Transfer learning/LinShu/DATA_linshu/000 Widefield"
base_dir = "/home/lyt//Data_attention/Transfer learning/LinShu/DATA_linshu/000 Widefield"
mouse_id = "A095"

stim_onset = 1.25 
stim_duration = 3
feedback_onset = 2.25
mouse_path = os.path.join(base_dir, mouse_id)
session_list = []

with os.scandir(mouse_path) as entries:
    for entry in entries:
        if entry.name.startswith('2025') and entry.is_dir():
            session_list.append(entry.name)

for session in session_list:
    run_tiff = False
    session_path = os.path.join(mouse_path, session)
    folders_470 = [f for f in os.listdir(session_path) 
               if os.path.isdir(os.path.join(session_path, f)) and f.endswith('470')]
    if len(folders_470)>0:
        target_folder = folders_470[0]
        target_path = os.path.join(session_path, target_folder)
        run_tiff = count_tiff(target_path)
        mean_tiff_files = (glob(os.path.join(session_path,'*dff_mean_Hit.tiff')) + glob(os.path.join(session_path,'*dff_mean_Hit.tif')))
    if session in ['20250730','20250801','20250807','20250808','20250811','20250813','20250911','20250930','20251001']:
        continue
    if run_tiff and len(mean_tiff_files)==0:
        print('Processing: '+session)
        rawPath = session_path
        
        if not os.path.exists(rawPath):
            print(f"The specified path does not exist: {rawPath}")
            continue
            
        timePath = os.path.join(rawPath, 'time')
        trial_path = os.path.join(timePath, 'trial_timestamp.npy')
        # if not os.path.exists(trial_path):
        #     config_path = os.path.join(os.path.abspath('.'),'config/A095_config.yaml')
        #     preprocess_mice(config_path)
        trial_timestamp = np.load(trial_path)
        stim_path = os.path.join(timePath, 'stim_timestamp.npy')
        stim_timestamp = np.load(stim_path)
        wf_path = os.path.join(timePath, 'widefield_timestamps_blue.npy')
        wf_timestamp = np.load(wf_path)
        stim_path = os.path.join(timePath, 'stim_timestamp.npy')
        stim_timestamp = np.load(stim_path)
        wf_path = os.path.join(timePath, 'widefield_timestamps_blue.npy')
        wf_timestamp = np.load(wf_path)
        wf_timestamp = wf_timestamp/1e3
        print(wf_timestamp.shape)
        plt.plot(trial_timestamp,'.')
        plt.show()
        plt.plot(stim_timestamp,'r.')
        plt.show()
        plt.plot(wf_timestamp,'.')
        plt.show()
        
        wf_trial_0_idx,wf_trial_0_diff = find_closest_indices(wf_timestamp, trial_timestamp[:,0])
        wf_trial_1_idx,wf_trial_1_diff = find_closest_indices(wf_timestamp, trial_timestamp[:,1])
        plt.plot(wf_trial_0_idx[1:]-wf_trial_0_idx[:-1],'.')
        plt.plot(wf_trial_1_idx[1:]-wf_trial_1_idx[:-1],'.')
        plt.show()
        wf_sf = 10 #Hz
        target_values = [5.2,5.3,9.7,9.8]
        trial_duration = (wf_trial_1_idx-wf_trial_0_idx)/wf_sf
        # trials_lossed_frames = np.where(~np.isin(trial_duration,target_values))[0]
        # for tr in trials_lossed_frames:
        #     _dff = dff[wf_trial_0_idx[tri]:wf_trial_1_idx[tri],:,:]
        #     _timestamp = wf_timestamp[wf_trial_0_idx[tri]:wf_trial_1_idx[tri]]
        #     _diffs = np.abs((_timestamp[-1]-_timestamp[0]) - target_values)
        #     _min_idx = np.argmin(_diffs)
        #     _timestamp_interp, _dff_interp = interpolate_xy(_timestamp,_dff,new_n=target_values[_min_idx]*wf_sf)
        
        
        _idx = np.logical_or(trial_duration==5.2,trial_duration==9.7)
        wf_trial_1_idx[_idx] = wf_trial_1_idx[_idx]+1
        plt.plot((wf_trial_1_idx-wf_trial_0_idx)/wf_sf,'.')
        plt.show()
        
        dff_text_path = os.path.join(rawPath, f'{mouse_id}_{session}_dff.tif')
        if os.path.exists(dff_text_path):
            dff = imread(dff_text_path)
        else:
        # mask save in ./rawPath/QualityControl/brain_mask.npy
            mask_path = os.path.join(rawPath, "QualityControl/brain_mask.npy")
            if os.path.exists(mask_path):
                mask = np.load(mask_path)
                # print(f"Mask shape: {mask.shape}")
                show_one_image(mask, title="Brain Mask")
            else:
                print(f"The specified mask path does not exist: {mask_path}")
                mask = None
    
            
            # n_preview = 300 # set to None to process all images, 
            # or set to an integer to limit the number of images processed for preview
            
            # if mask is not None, apply the mask to the images
            stack, files_470 = images2tiff_two_channel(rawPath,
                mask=mask)
            print("The shape of the processed images:", stack.shape)
            
            # visualize the first image of each channel
            images = list(stack[0])
            show_images(images, titles=['470 channel', '405 channel'])
            
            
            stack_mean = np.mean(stack, axis=(2, 3))
            print("The shape of the stack_mean:", stack_mean.shape)
            outlier_index_470, outlier_index_405 = detect_lum_outlier(stack_mean, 
                threshold=6, plot=True)
            
            stack_corrected = correct_lum_outlier(stack, outlier_index_470,
                outlier_index_405, plot=False)
            
            del stack
            #% calculate delta_f/f for 470 channel
            # the 'delta_ff_3d' is time consuming, optimize 'cal_base' may help
            
            
            # stack_470_dff = delta_ff_3d(stack_corrected[:, 0, :, :], win=150)
            # # print("The shape of stack_470_dff:", stack_470_dff.shape)
            
            # stack_405_dff = delta_ff_3d(stack_corrected[:, 1, :, :], win=150)
            # # print("The shape of stack_405_dff:", stack_405_dff.shape)
            
            
            
            print('Calculating dF/F0...')
            dff = delta_ff_3d_2ch(stack_corrected[:, 0, :, :],stack_corrected[:, 1, :, :], win=150)
            
            del stack_corrected
            #%
            
            # dff = stack_470_dff - stack_405_dff
            # Scale the dff to range of 16 bit
            dff_min = dff.min()
            dff_max = dff.max()
            dff_range = dff_max - dff_min
            
            if dff_range > 0:
                scale_factor = 0.9*65535/dff_range
            else:
                scal_factor = 1
                
            metadata = {'dff_min':dff_min,
                        'dff_max':dff_max,
                        'scale_factor':scale_factor}
            
            dff_uint16 = np.empty(dff.shape, dtype=np.uint16)
            
            for i in tqdm(range(dff.shape[0])):  # 遍历 29,927 帧
                scaled_data = scale_factor * (dff[i] - dff_min)
                dff_uint16[i] = scaled_data.astype(np.uint16)
                
            dff = dff_uint16
            del dff_uint16
                    
            dff_text = np.copy(dff)
            for trial,i in enumerate(wf_trial_0_idx):
                if i <dff_text.shape[0]:
                    dff_text[i,:,:] = add_text_to_image(dff[i,:,:], trial+1, status="on")
            for trial,i in enumerate(wf_trial_1_idx):
                if i <dff_text.shape[0]:
                    dff_text[i,:,:] = add_text_to_image(dff[i,:,:], trial+1, status="off")
            
            
            print('Text added.')
            # save the dff_scaled as tiff file
            save_path = os.path.join(rawPath, f'{mouse_id}_{session}_dff.tif')
            imwrite(save_path, dff_text.astype(np.uint16),metadata=metadata, imagej=True)
            
            
            meta_file = save_path.replace('.tif','_metadata.npy')
            np.save(meta_file, metadata, allow_pickle=True) 
            
            del dff_text
        
        # dff = 65535 * (dff - dff_min) / (dff_max - dff_min)
        
        #%
        
       
        # plt.plot(wf_timestamp[18001:wf_trial_1_idx[-1]+6]-wf_timestamp[18000:wf_trial_1_idx[-1]+5],'.')
        # plt.show()
        # acutal trial duration is 5.25 s or  9.75 s
        
        # plt.plot(wf_timestamp[wf_trial_0_idx[280:]],'.')
        # plt.plot(wf_timestamp[wf_trial_1_idx[280:]],'.')
        # plt.show()
        
        # plt.plot(wf_trial_0_idx[280:],'.')
        # plt.plot(wf_trial_1_idx[280:],'.')
        # plt.show()
        
        
        

        #%
        # same averaged images: across all, Hit, CR, FA, Miss trials
        trial_type_path = os.path.join(timePath, 'trial_type.npy')
        if not os.path.exists(trial_type_path):
            trial_type = load_trial_type(rawPath, timePath)
        else:    
            trial_type = np.load(trial_type_path)
        plt.plot(trial_type,'.')
        n_trials = wf_trial_0_idx.size
        dff_mean_all = compute_trial_mean(dff, wf_timestamp, wf_trial_0_idx,wf_trial_1_idx)
        save_path = os.path.join(rawPath, f'{mouse_id}_{session}_dff_mean_all.tif')
        imwrite(save_path, dff_mean_all.astype(np.uint16), imagej=True)
        print('The dff tiff file saved to:', save_path)
        
        
        type_list = ['Hit','Miss', 'FA','CR']
        for i in range(1,5):
            dff_mean = compute_trial_mean(dff,wf_timestamp,wf_trial_0_idx[trial_type==i],wf_trial_1_idx[trial_type==i])
            save_path = os.path.join(rawPath, f'{mouse_id}_{session}_dff_mean_{type_list[i-1]}.tif')
            imwrite(save_path, dff_mean.astype(np.uint16), imagej=True)
            print('The dff tiff file saved to:', save_path)
        # pre_process_wf(session_path)
    else:
        if len(mean_tiff_files)>0:
            print('Already processed: '+session)
        else:
            print('Cannot be processed: '+session)
        
    
#%%

def load_trial_type(rawPath, timePath):
    """
    从指定路径的日志文件中读取 trial_type 信息，并保存为 .npy 文件到 timePath。

    参数:
        rawPath (str): 存放 log 文件的文件夹路径。
        timePath (str): 保存输出 .npy 文件的目标路径。

    返回:
        list[int]: trial_type 列表
    """
    # 匹配路径下的 log 文件
    txt_path = glob(pjoin(rawPath, '*log_*.txt'))
    if not txt_path:
        raise FileNotFoundError(f"未在路径 {rawPath} 下找到匹配的 log 文件。")

    trial_type = None
    # 打开第一个匹配的文件
    with open(txt_path[0], 'r') as f:
        for line in f:
            if line.startswith("Data:"):
                values = line.split(":", 1)[1].strip()
                values = values.strip("[]")
                trial_type = [int(x) for x in values.split()]
                break

    if trial_type is None:
        raise ValueError(f"文件 {txt_path[0]} 中未找到以 'Data:' 开头的行。")

    # 确保保存目录存在
    os.makedirs(timePath, exist_ok=True)

    # 保存为 .npy 文件
    base_name = os.path.splitext(os.path.basename(txt_path[0]))[0]
    save_path = pjoin(timePath, f"trial_type.npy")
    np.save(save_path, np.array(trial_type))

    print(f"trial_type 已保存到 {save_path}")
    return trial_type
for session in session_list:
    session_path = os.path.join(mouse_path, session)
    rawPath = session_path
    timePath = os.path.join(rawPath, 'time')
    trial_type_path = pjoin(timePath, f"trial_type.npy")
    run_tiff = False
    session_path = os.path.join(mouse_path, session)
    folders_470 = [f for f in os.listdir(session_path) 
               if os.path.isdir(os.path.join(session_path, f)) and f.endswith('470')]
    if len(folders_470)>0:
        target_folder = folders_470[0]
        target_path = os.path.join(session_path, target_folder)
        run_tiff = count_tiff(target_path)
        mean_tiff_files = (glob(os.path.join(session_path,'*dff.tiff')) + glob(os.path.join(session_path,'*dff.tif')))
    if run_tiff:
        if not os.path.exists(trial_type_path):
            trial_type = load_trial_type(rawPath, timePath)
            print(session)
        else:    
            trial_type = np.load(trial_type_path)
        plt.plot(trial_type,'.')
        
#%%
