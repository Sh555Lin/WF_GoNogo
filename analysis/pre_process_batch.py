#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 18 09:19:56 2025

@author: yatangli
"""

from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import os, glob
import pandas as pd
import sys
from PIL import Image, ImageDraw, ImageFont
from tifffile import imwrite
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
    trial_timestamp = np.load(trial_path)
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
    
    wf_trial_0_idx = find_closest_indices(wf_timestamp, trial_timestamp[:,0])
    wf_trial_1_idx = find_closest_indices(wf_timestamp, trial_timestamp[:,1])
    wf_sf = 10 #Hz
    trial_duration = (wf_trial_1_idx-wf_trial_0_idx)/wf_sf
    _idx = np.logical_or(trial_duration==5.2,trial_duration==9.7)
    wf_trial_1_idx[_idx] = wf_trial_1_idx[_idx]+1
    plt.plot((wf_trial_1_idx-wf_trial_0_idx)/wf_sf,'.')
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
base_dir = "/Volumes/Data_attention/Transfer learning/LinShu/DATA_linshu/000 Widefield"
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
        mean_tiff_files = (glob.glob(os.path.join(session_path,'*dff.tiff')) + glob.glob(os.path.join(session_path,'*dff.tif')))
    if run_tiff and len(mean_tiff_files)==0:
        pre_process_wf(session_path)
    
        