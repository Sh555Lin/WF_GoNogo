from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import scipy
import yaml
from PIL import Image, ImageDraw, ImageFont
from glob import glob
from os.path import join as pjoin
from scipy import interpolate

import os, pickle
import pandas as pd
import sys
from tifffile import imwrite, imread
import pathlib
from skimage.draw import polygon
import warnings
from scipy.stats import f
import json
from scipy.stats import norm

import h5py

def cal_base(x, n_bins=10, thr=1/3):
    """
    Calculate the base of the data based on the histogram of the data.
    x: 1D array
    n_bins: number of bins
    thr: threshold of the bin count to determine the base
    """
    count, bins = np.histogram(x, n_bins)
    count_max = count.max()
    for i in range(count.size):
        if count[i] > count_max*thr:
            base = bins[i+1]
            break
    return base

def correct_lum_outlier(images, outlier_index_470, outlier_index_405,
    plot=True):
    '''
    Correct the luminance outliers frames in the merged tiff file.
    outlier_index_xxx: list of tuples, each tuple contains the start and end
    index of the outlier segments.
    '''
    if (outlier_index_470 is not None) or (outlier_index_405 is not None):

        if outlier_index_470 is not None:
            for i in range(len(outlier_index_470)):
                start, end = outlier_index_470[i]
                if end+1<images.shape[0]:
                    images[start:end+1, 0, :, :] = 0.5*images[start-1, 0, :, :] + \
                        0.5*images[end+1, 0, :, :]

        if outlier_index_405 is not None:
            for i in range(len(outlier_index_405)):
                start, end = outlier_index_405[i]
                images[start:end+1, 1, :, :] = 0.5*images[start-1, 1, :, :] + \
                    0.5*images[end+1, 1, :, :]
    else:
        print('No luminance outliers frames to correct!')

    if plot:
        print('Checking the corrected images...')
        print('Calulating the mean values of corrected images...')
        mean_values_470 = images[:, 0, :, :].mean(axis=(1,2))
        mean_values_405 = images[:, 1, :, :].mean(axis=(1,2))
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.plot(mean_values_470, label='470', color='r')
        ax.plot(mean_values_405, label='405', color='k')
        ax.legend()
        plt.title('correct_lum_outlier')
        plt.show()

    return images

def delta_ff_1d(data, win=150):
    '''
    calculate the delta_f/f
    data: the raw data, 1D array
    '''
    trend = moving_average(data,window=win)
    detrended = data - trend
    baseline = cal_base(detrended) + trend
    d_ff = (data - baseline) / baseline
    return d_ff

def delta_ff_3d(data, win=150):
    """
    Calculate the delta_f/f of a 3D array.
    data: 3D array, shape (n_frames, pixels, pixels)
    win: window size
    """
    # print("delta_ff_3d: Calculating delta_f/f ...")
    d_ff = np.zeros(data.shape)
    for i in tqdm(range(data.shape[1])):
        for j in range(data.shape[2]):
            d_ff[:, i, j] = delta_ff_1d(data[:, i, j], win=win)
    return d_ff

def delta_ff_3d_2ch(data,data_ref, win=150):
    """
    Calculate the delta_f/f of a 3D array.
    data: 3D array, shape (n_frames, pixels, pixels)
    win: window size
    """
    # print("delta_ff_3d: Calculating delta_f/f ...")
    dff = np.zeros(data.shape,dtype=np.float32)
    for i in tqdm(range(data.shape[1])):
        for j in range(data.shape[2]):
            dff[:, i, j] = delta_ff_1d(data[:, i, j], win=win)-delta_ff_1d(data_ref[:, i, j], win=win)
    return dff

def detect_lum_outlier(mean_values, threshold=6, plot=True):
    '''
    Detect outliers in the mean_values array.
    Parameters:
        mean_values: np.array, shape=(nframes, 2), mean values of 470 and 405 channels
        threshold: float, threshold for detecting outliers
        plot: bool, whether to plot the detrended z-scores
    Returns:
        outlier_index: list of two lists, each containing tuples of (start, end) indices of outliers
    '''
    from scipy.stats import zscore

    z_scores = zscore(mean_values, axis=0)
    detrended = abs(detrend(z_scores, axis=0, window=300))
    # detecting luminance outliers of 470 channel
    _index = np.where(detrended[:, 0] > threshold)[0]
    # print(_index)
    # split the index into continuous segments
    _segments_index = np.where(np.diff(_index) != 1)[0] + 1
    _segments = np.split(_index, _segments_index)
    outlier_lum_idx_470 = []
    for k, g in enumerate(_segments):
        # Frames before and after the detected outliers are also considered as
        # outliers,
        # Output index is the start and end index of the outliers,
        # Note the frame of the end index is normal, so add 2 to the end index.
        outlier_lum_idx_470.append((g[0]-1, g[-1]+2)) if len(g) > 0 else None

    # detecting luminance outliers of 405 channel
    _index = np.where(detrended[:, 1] > threshold)[0]
    # split the index into continuous segments
    _segments_index = np.where(np.diff(_index) != 1)[0] + 1
    _segments = np.split(_index, _segments_index)
    outlier_lum_idx_405 = []
    for k, g in enumerate(_segments):
        outlier_lum_idx_405.append((g[0]-1, g[-1]+2)) if len(g) > 0 else None
        
    outlier_index = [outlier_lum_idx_470, outlier_lum_idx_405]

    if plot:
        fig, axis = plt.subplots(nrows=2, figsize=(10, 3), sharex=True)
        axis[0].plot(detrended[:, 0], label='470', color='red')
        axis[1].plot(detrended[:, 1], label='405', color='black')
        # axis[0].set_xlim(0, len(mean_values))
        # axis[1].set_xlim(0, len(mean_values))
        if len(outlier_lum_idx_470) > 0 or len(outlier_lum_idx_405) > 0:
            axis[0].hlines(threshold, 0, len(mean_values), color='g',
                linestyle='--', label='Threshold')
            axis[1].hlines(threshold, 0, len(mean_values), color='g',
                linestyle='--', label='Threshold')
        axis[0].legend()
        axis[1].legend()
        plt.suptitle('detect_lum_outlier')
    
    if len(outlier_lum_idx_470) > 0 or len(outlier_lum_idx_405) > 0:
        print('detect_lum_outlier: There are dim outliers frames detected!')
    else:
        print('detect_lum_outlier: No dim outliers frames detected!')

    return outlier_index

def detrend(data, axis=0, window=300):
    # from lib.utils import moving_average
    '''
    data: numpy.ndarray, shape (n_frames, n_rois)
    window: int
    '''
    # data_mean = np.mean(data, axis=0)
    average = moving_average(data, window=window, axis=axis)
    print(average.shape)
    average_mean = np.mean(average, axis=axis) 
    print(average_mean.shape)

    detrended = (data - average) + average_mean
    return detrended

def rotate_crop_array(array, angle, left, top, width, height):
    '''
    Rotate the image by the angle and crop the image with the given coordinates
    Args:
        array: numpy array, the image to be rotated and cropped, 2D or 3D. 
        if 3D, the first dimension is the number of images
        angle: float, the rotation angle in degrees. Positive, counterclockwise;
        Negative, clockwise.
        left: int, the x coordinate of the top-left corner of the cropped image
        top: int, the y coordinate of the top-left corner of the cropped image
        width: int, the width of the cropped image
        height: int, the height of the cropped image
    '''

    from scipy.ndimage import rotate
    if not angle == 0:
        if array.ndim == 2:
            rotated = rotate(array, angle, reshape=False)
        if array.ndim == 3:
            print('Rotating the images...')
            print('This may take a while..., please wait')
            rotated = rotate(array, angle, axes=(2, 1), reshape=False)
        print(rotated.shape)
    else:
        rotated = array

    if left+width > rotated.shape[-2] or top+height > rotated.shape[-1]:
        print(left+width, rotated.shape[-2], top+height, rotated.shape[-1])
        print("The crop area is out of the image")
        return None
    else:
        if rotated.ndim == 3:
            print('Cropping the images...')
            cropped = rotated[:, top:top+height, left:left+width]
        if rotated.ndim == 2:
            cropped = rotated[top:top+height, left:left+width]
        return cropped

def images2tiff_two_channel(folder, n_preview=None,
    crop_params=None, mask=None):
    '''
    convert the filtered images to one tiff file
    folder: the folder where the images are saved, may contain multiple folders
    save_path: the path to save the tiff file
    n_preview: if not None, only convert the first n_preview images
    crop_params: if not None, a dict containing the crop parameters
        {'crop': (left, top, width, height), 'rotate_angle': angle}
    mask: if not None, a binary mask to apply to the images
    2 channels are assumed, the first channel is 470, the second channel is 405
    470 and 405 images are assumed to be in separate folders, named as *-470
    and *-405, respectively.
    470 and 405 images are assumed to be named as 1.tif, 2.tif, ..., n.tif
    470 and 405 images are assumed to be in the same number

    The output tiff file is a 4D array, with shape (n_images, 2, height, width)
    '''
    
    import glob
    from natsort import natsorted
    from tifffile import imread

    if not os.path.exists(folder):
        print(f'Folder {folder} does not exist!')
        return None, None

    files_470 = glob.glob(os.path.join(folder, '*-470/*.tif'))
    files_470 = natsorted(files_470)
    n_files_470 = int(os.path.basename(files_470[-1]).split('.')[0])
    print('The number of 470 images:', n_files_470)

    files_405 = glob.glob(os.path.join(folder, '*-405/*.tif'))
    files_405 = natsorted(files_405)
    n_files_405 = int(os.path.basename(files_405[-1]).split('.')[0])
    print('The number of 405 images:', n_files_405)
    _image = imread(files_470[0])
    image_size = _image.shape

    n_tif = min(n_files_470, n_files_405) if n_preview is None else \
        min(n_files_470, n_files_405, n_preview)
    
    if crop_params is None:
        all_images = np.zeros((n_tif, 2, image_size[0], image_size[1]),
            dtype=_image.dtype)
    else:
        left, top, width, height = crop_params['crop']
        angle = crop_params['rotate_angle']
        all_images = np.zeros((n_tif, 2, height, width),
            dtype=_image.dtype)

    for i in tqdm(range(n_tif)):
        
        _path_470 = glob.glob(os.path.join(folder, f'*-470/{i+1}.tif'))
        # print(_path_470)
        _path_405 = glob.glob(os.path.join(folder, f'*-405/{i+1}.tif'))
        
        # skip_iteration = False
        
        if len(_path_470) == 1:
            try:
                image_470 = imread(_path_470[0])    
                image_470 = image_470 * mask if mask is not None else image_470
                if crop_params is not None:
                    image_470 = rotate_crop_array(image_470, angle=angle,
                        left=left, top=top, width=width, height=height)
                all_images[i, 0, :, :] = image_470
            except Exception as e:
                error_msg = f"Error reading 470 channel file {_path_470[0]}: {str(e)}"
                print(f"Warning: {error_msg}")
                continue

        else:
            print(f'File {i}.tif not found in 470 channel!')
    
    for i in tqdm(range(n_tif)):
        if len(_path_405) == 1:
            try:
                image_405 = imread(_path_405[0])
                image_405 = image_405 * mask if mask is not None else image_405
                if crop_params is not None:
                    image_405 = rotate_crop_array(image_405, angle=angle,
                        left=left, top=top, width=width, height=height)
                all_images[i, 1, :, :] = image_405
            except Exception as e:
                error_msg = f"Error reading 405 channel file {_path_405[0]}: {str(e)}"
                print(f"Warning: {error_msg}")
        else:
            print(f'File {i}.tif not found in 405 channel!')

    # print('The file saved to:', save_path)
    # imwrite(save_path, all_images, imagej=True)
    return all_images, files_470

def moving_average(data, window=300, axis=0):
    '''
    data: numpy.ndarray, the first dimension is samples
    window: int
    '''
    from scipy.ndimage import uniform_filter1d

    return uniform_filter1d(data, size=window, axis=axis)

def show_images(images, vlim=None, title=None, identical_colorbar=False,
        **kwargs):
    '''
    Show a list of images in a grid.
    images: list
    '''

    nrows = 1
    ncols = len(images)

    if vlim is None:
        vmin = np.array([image.min() for image in images]).min()
        vmax = np.array([image.max() for image in images]).max()
    else:
        vmin, vmax = vlim

    figsize = kwargs.get('figsize', (3*ncols, 3*nrows))

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
        layout='constrained')
    cmap = kwargs.get('cmap', 'hot')
    for i, ax in enumerate(axes):
        if identical_colorbar:
            cb = ax.imshow(images[i], cmap=cmap, clim=(vmin, vmax))
        else:
            cb = ax.imshow(images[i], cmap=cmap)
        ax.grid()

    if 'titles' in kwargs:
        for i, ax in enumerate(axes):
            ax.set_title(kwargs['titles'][i])
    if identical_colorbar:
        fig.colorbar(cb, ax=axes, orientation='vertical', fraction=0.03,
            shrink=0.5)
    if 'suptitle' in kwargs:
        fig.suptitle(kwargs['suptitle'])
    plt.show()

def plot_traces(data, data_rate=None, figsize=(15, 5), title=None,
    xlim=None, ylim=None, labels=None, **kwargs):
    '''
    Plot the traces of the data.
    data: np.array, shape=(nframes, ntrials)
    data_rate: int, the rate of the data, default is None
    labels: None or list of str, the labels of the traces
    '''
    if 'offset' in kwargs:
        data = data + np.arange(data.shape[1]) * kwargs['offset']
    fontsize = 7
    plt.rc('axes', labelsize=fontsize)
    plt.rc('xtick', labelsize=fontsize)
    plt.rc('ytick', labelsize=fontsize)

    prop_cycle = plt.rcParams['axes.prop_cycle']
    alpha = kwargs.get('alpha', 1.0)
    colors = kwargs.get('colors', prop_cycle.by_key()['color'])

    n_colors = len(colors)

    fig, ax = plt.subplots(figsize=figsize)

    if data_rate is not None:
        x = np.arange(data.shape[0]) / data_rate
    else:
        x = np.arange(data.shape[0])

    for i in range(data.shape[1]):
        if labels is not None:
            ax.plot(x, data[:, i], label=labels[i], lw=1,
                color=colors[i % n_colors], alpha=alpha)
        else:
            ax.plot(x, data[:, i], lw=1, color=colors[i % n_colors],
                alpha=alpha)

    ax.spines[['top', 'right']].set_visible(False)

    if data_rate is not None:
        ax.set_xlabel('Time (s)')
    else:
        if 'xlabel' in kwargs:
            ax.set_xlabel(kwargs['xlabel'])

    if 'vlines' in kwargs:
        for vline in kwargs['vlines']:
            ax.axvline(x=vline, color='r', linestyle='--')

    if 'hlines' in kwargs:
        for i, hline in enumerate(kwargs['hlines']):
            ax.axhline(y=hline, color=colors[i])

    ax.grid(True, linestyle='--') if 'grid' in kwargs else None
    ax.set_title(title, fontsize=8) if title is not None else None
    ax.set_ylabel(kwargs['ylabel']) if 'ylabel' in kwargs else None
    ax.set_ylim(bottom=ylim[0], top=ylim[1]) if ylim is not None else None

    if xlim is not None:
        ax.set_xlim(left=xlim[0], right=xlim[1])
    else:
        ax.set_xlim(left=-x[-1]*0.01, right=x[-1]*1.01)
    if labels is not None:
        ax.legend(loc='upper right', frameon=False, fontsize=5)
    
    plt.show()

def show_one_image(image: np.ndarray, cmap='gray', colorbar=False,
    figsize=(5, 5), grid=False, title=None, vmin=None, vmax=None,
    **kwargs) -> None:
    '''
    show one image
    '''
    if 'ax' in kwargs:
        ax = kwargs['ax']
    else:
        fig, ax = plt.subplots(figsize=figsize)
    if vmin is not None and vmax is not None:
        # print(f'show_one_image:vmin={vmin}, vmax={vmax}')
        cb = ax.imshow(image, cmap=cmap, aspect='equal', vmin=vmin, vmax=vmax)
    else:
        cb = ax.imshow(image, cmap=cmap, aspect='equal')

    if 'ROI' in kwargs:
        from matplotlib.patches import Rectangle
        ROI = kwargs['ROI']
        xy = (ROI[0], ROI[1])
        width, height = ROI[2], ROI[3]
        rect = Rectangle(xy, width, height, linewidth=2, edgecolor='r',
            facecolor='none')
        ax.add_patch(rect)
    
    if ('ticks' in kwargs) and (kwargs['ticks'] == False):
        ax.set_xticks([])
        ax.set_yticks([])

    if title is not None:
        ax.set_title(title)
        
    ax.set_xlabel(kwargs.get('xlabel', None))
    ax.set_ylabel(kwargs.get('ylabel', None))
    ax.grid() if grid else None
    plt.colorbar(cb, shrink=0.8) if colorbar else None
    plt.show() if 'ax' not in kwargs else None
    
    
def load_config(config_path):
    """load the YAML config"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

# def preprocess_mice(config_path):
#     config = load_config(config_path)
#     for session in config["sessions"]:
#         load_timelite(config, session)
#     print(f"all done")
    
def find_closest_indices(x, y):
    """
    For each element in y, find the index in x with the closest value.
    
    Parameters:
    x: numpy array of size n
    y: numpy array of size m (m < n)
    
    Returns:
    numpy array of indices in x (size m)
    """
    indices = []
    diffs = []
    
    for value in y:
        # Calculate absolute differences
        differences = np.abs(x - value)
        # Find index of minimum difference
        closest_idx = np.argmin(differences)
        indices.append(closest_idx)
        diffs.append(np.min(differences))
    
    return np.array(indices),np.array(diffs)


def add_trial_text_clean(image_array, trial_num, status="on"):
    """
    Add text without affecting other bright pixels.
    
    Parameters:
    image_array: (512, 512) uint16 array
    trial_num: int in [1, 500]
    status: "on" or "off"
    
    Returns:
    (512, 512) uint16 array with clean text overlay
    """
    # Validate inputs
    if not 1 <= trial_num <= 500:
        raise ValueError("trial_num must be between 1 and 500")
    if status not in ["on", "off"]:
        raise ValueError("status must be 'on' or 'off'")
    
    text = f"trial {trial_num} {status}"
    result = image_array.copy()
    
    # Convert to PIL (16-bit)
    pil_image = Image.fromarray(result, mode='I;16')
    
    # Create a completely black image for text mask
    black_image = Image.new('L', (512, 512), 0)
    draw_black = ImageDraw.Draw(black_image)
    
    # Create a completely white image for text mask
    white_image = Image.new('L', (512, 512), 255)
    draw_white = ImageDraw.Draw(white_image)
    
    # Try to load font
    try:
        font = ImageFont.truetype("arial.ttf", 30)
    except:
        font = ImageFont.load_default()
    
    # Draw text on both images
    position = (10, 10)
    draw_black.text(position, text, fill=255, font=font)  # White text on black
    draw_white.text(position, text, fill=0, font=font)    # Black text on white
    
    # Convert to numpy arrays
    black_array = np.array(black_image)
    white_array = np.array(white_image)
    
    # Get exact text mask: pixels that are white in black image AND black in white image
    text_mask = (black_array == 255) & (white_array == 0)
    
    # Apply text to original image (set to max 16-bit value)
    result[text_mask] = 65535
    
    return result


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

def interpolate_xy(x, y, new_n=None):
    """
    Interpolate x (timestamps) and y (3D array) with accurate first and last timestamps.
    
    Parameters:
    x: array of shape (n,) - timestamps
    y: array of shape (n, 512, 512) - corresponding data
    new_n: int - desired number of interpolated samples (default: n*2)
    
    Returns:
    x_interp: interpolated timestamps
    y_interp: interpolated 3D array
    """
    n = len(x)
    new_n = int(new_n)
    if new_n is None:
        new_n = n  # Default: double the samples
    
    # Create new equally spaced timestamps using accurate first/last
    x_interp = np.linspace(x[0], x[-1], new_n)
    
    # Initialize output array
    y_interp = np.zeros((new_n, 512, 512))
    
    # Interpolate each pixel independently
    for i in range(512):
        for j in range(512):
            # Linear interpolation for each pixel
            interp_func = interpolate.interp1d(x, y[:, i, j], kind='linear', 
                                               fill_value='extrapolate')
            y_interp[:, i, j] = interp_func(x_interp)
    
    return x_interp, y_interp

def compute_trial_mean(dff, wf_timestamp,idx_onset, idx_offset):
    """
    Vectorized version using padding for speed.
    """
    
    # Calculate trial lengths
    trial_lengths = idx_offset - idx_onset
    n_trials = len(idx_onset)
    # Find the two possible lengths
    target_values = [5.2,5.3,9.7,9.8]
    wf_sf = 10 #Hz
    unique_lengths = np.intersect1d(np.unique(trial_lengths),np.array(target_values)*10).astype(int)
    
    # Choose the shorter length
    n_frames= np.min(unique_lengths)
        
    print(n_frames)
   

    _, x, y = dff.shape
    
    # Pre-allocate array for all trials
    # Shape: [n_trials, n_frames, x, y]
    all_trials = np.full((n_trials, n_frames, x, y), np.nan, dtype=np.float32)
    
   
    # Extract and pad each trial
    for i in tqdm(range(n_trials)):
        start = idx_onset[i]
        end = start+n_frames
        # if end>idx_offset[i]:
        #     print(i)
        #     _dff = dff[start:idx_offset[i]]
        #     _timestamp = wf_timestamp[start:idx_offset[i]]
        #     _diffs = np.abs((_timestamp[-1]-_timestamp[0]) - target_values)
        #     _min_idx = np.argmin(_diffs)
        #     _timestamp_interp, _dff_interp = interpolate_xy(_timestamp,_dff,new_n=target_values[_min_idx]*wf_sf)
        #     trial_data = _dff_interp
        # else:
        #     trial_data = dff[start:end] 
        # print(start)
        if end>idx_offset[i]:
            print('')
            print(str(i)+'th trial has lost frames!')
            continue
        trial_data = dff[start:end] 
        trial_len = trial_data.shape[0]
        # Pad to max length
        all_trials[i, :trial_len] = trial_data
    # Compute nanmean across trials
    dff_mean = np.nanmean(all_trials, axis=0)
    return dff_mean


def add_text_to_image(image_array, trial_num, status="on"):
    """
    Add text without affecting other bright pixels.
    
    Parameters:
    image_array: (512, 512) uint16 array
    trial_num: int in [1, 500]
    status: "on" or "off"
    
    Returns:
    (512, 512) uint16 array with clean text overlay
    """
     # Validate
    if not 1 <= trial_num <= 500:
        raise ValueError("trial_num must be between 1 and 500")
    if status not in ["on", "off"]:
        raise ValueError("status must be 'on' or 'off'")
    
    text = f"Trial {trial_num} {status}"
    result = image_array.copy()
    
    # Create a transparent image for text
    text_image = Image.new('RGBA', (512, 512), (0, 0, 0, 0))
    draw = ImageDraw.Draw(text_image)
    
    # Try to load font
    # Try to load a nice font
    font_size = 30
    try:
        # Try common font paths
        font_paths = [
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
            "/System/Library/Fonts/Helvetica.ttc",
            "C:/Windows/Fonts/arial.ttf",
            "arial.ttf"
        ]
        
        font = None
        for path in font_paths:
            if os.path.exists(path):
                try:
                    font = ImageFont.truetype(path, font_size)
                    break
                except:
                    continue
        
        if font is None:
            font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    
    # Draw text in white with full opacity
    position = (10, 10)
    draw.text(position, text, fill=(255, 255, 255, 255), font=font)
    
    # Convert to numpy array
    text_array = np.array(text_image)
    
    # Extract alpha channel (where text is)
    alpha_channel = text_array[:, :, 3]  # Alpha is the 4th channel
    
    # Create mask where alpha > 0 (text pixels)
    text_mask = alpha_channel > 0
    
    # Apply text to original image
    result[text_mask] = 65535
    
    return result


def count_tiff(target_path, threshold=1000):
    """
    Fastest method - counts both .tiff and .tif files
    threshold: if provided, returns as soon as count exceeds threshold
    """
    count = 0
    
    with os.scandir(target_path) as entries:
        for entry in entries:
            if entry.is_file():
                name_lower = entry.name.lower()
                if name_lower.endswith(('.tiff', '.tif')):
                    count += 1
                    if threshold is not None and count > threshold:
                        return True  # Early exit
                    
    return count>threshold 


def savefig(file_save):
    plt.savefig(file_save+'.png',bbox_inches='tight')
    plt.savefig(file_save+'.svg',bbox_inches='tight')
    
    
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

    summary = {
        "hit_rate": hit_rate_all,
        "fa_rate": fa_rate_all,
        "dprime": dprime_all,
        "n_trials": n_trials
    }

    return dprimes, summary
    
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


def compute_daily_summary_tocsv(hdf5_path,lick_frequency_csv,save_dir=None):
    
    all_summaries = []
    with h5py.File(hdf5_path, 'r') as f:
        mouse_id = f.attrs.get('mouse_id')
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


def plot_individual_metrics_from_csv(mouseid,csv_path,save_dir):
    df = pd.read_csv(csv_path)
    
    metrics = ['dprime', 'accuracy', 'hit_rate', 'fa_rate']
    titles = ['d-prime', 'Accuracy', 'Hit Rate', 'False Alarm Rate']
    y_labels = ['d-prime', 'Accuracy', 'Hit Rate', 'False Alarm Rate']
    
    states = ['Condition', 'Discrimination', 'Switch']
    state_colors = {'Condition': '#FFE5E5', 'Discrimination': '#E5F7FF', 'Switch': '#F0FFE5'}
    days_per_state={}
    for state in states:
        if state in df['state'].unique():
            days_per_state[state] = len(df[df['state']==state])
    x_start = 0
    state_x_ranges = {}
    for state in states:
        if state in days_per_state:
            state_x_ranges[state] = (x_start, x_start + days_per_state[state])
            x_start += days_per_state[state]
    for metric, title, y_label in zip(metrics, titles, y_labels):
        plt.figure(figsize=(30,6))
        ax = plt.gca()

        mouse_data = df.copy()
        mouse_data = mouse_data.sort_values('date')
        mouse_data['day_in_state'] = mouse_data.groupby('state').cumcount()
            
        x_points = []
        y_points = []
        state_breaks = []

        for state in states:
                if state in mouse_data['state'].unique():
                    state_data = mouse_data[mouse_data['state']==state].sort_values('date')
                    state_x = state_x_ranges[state][0] + state_data['day_in_state'].values
                    state_y = state_data[metric].values
                    
                    if len(state_x) > 1:
                        ax.plot(state_x, state_y, 'o-',
                               linewidth=2, markersize=4)  # 移除 label=mouse
                    elif len(state_x) == 1:
                        ax.plot(state_x, state_y, 'o',
                               markersize=6)  # 移除 label=mouse
                    
                    x_points.extend(state_x)
                    y_points.extend(state_y)
                    
                    if len(state_data) > 0 and state != states[-1]:
                        state_breaks.append(state_x[-1])
            
        for break_x in state_breaks:
                idx = np.searchsorted(x_points, break_x)
                if idx < len(x_points) - 1:
                    ax.plot([x_points[idx], x_points[idx+1]], [y_points[idx], y_points[idx+1]], 
                           '--', alpha=0.5, linewidth=1)
        
        for state, (x_start, x_end) in state_x_ranges.items():
            ax.axvspan(x_start, x_end, alpha=0.2, color=state_colors[state], zorder=-1)
        if metric in ['hit_rate', 'fa_rate','miss_rate','cr_rate']:
            ax.set_ylim([-0.05, 1.1])

        ax.set_title(f'{mouseid} {title} Across Training States', fontsize=16, fontweight='bold')
        ax.set_xlabel('Training Days (aligned by state)', fontsize=14)
        ax.set_ylabel(y_label, fontsize=14)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        ax.set_xticks([])
        for state, (x_start, x_end) in state_x_ranges.items():
            ax.axvline(x=x_start, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
            ax.axvline(x=x_end, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
        
        # 创建状态图例
        state_handles = [plt.Rectangle((0,0), 1, 1, color=state_colors[state], alpha=0.2) 
                         for state in state_x_ranges.keys()]
        ax.legend(state_handles, state_x_ranges.keys(), title='State', 
                  loc='upper left', bbox_to_anchor=(1.02, 0.6), 
                  borderaxespad=0., fontsize=10, frameon=True, 
                  fancybox=True, shadow=True)
        
        plt.tight_layout(rect=[0, 0, 0.85, 1])  # 调整右侧留出空间给图例
        
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f'{metric}_of_{mouseid}_across_day.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.savefig(f'{metric}_of_{mouseid}_across_day.png', dpi=300, bbox_inches='tight')
            print(f"Plot saved to {metric}_across_day.png")
        
        plt.show()
        
        
def load_pickle_safely(filename):
    """
    Safely load pickle files - handles both file paths and file objects
    """
    # If filename is already a file object, use it directly
    if hasattr(filename, 'read'):
        return pickle.load(filename)
    
    # If it's a string/path, open it
    else:
        with open(filename, 'rb') as f:
            return pickle.load(f)
    
    
def norm_x(x):
    x_min = np.nanmin(x)
    x_max = np.nanmax(x)
    x_norm = (x-x_min)/(x_max-x_min)
    return x_norm

def extract_region_mean_response(x, y, dff):
    """
    Version using scikit-image for polygon operations.
    """
    
    # Ensure inputs are numpy arrays
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Get polygon coordinates (ensure they're integers)
    x_int = np.round(x).astype(int)
    y_int = np.round(y).astype(int)
    
    # Get dimensions
    n_frames, height, width = dff.shape  # (53, 512, 512)
    
    # Clip coordinates to image bounds
    x_int = np.clip(x_int, 0, width - 1)      # 0 to 511
    y_int = np.clip(y_int, 0, height - 1)     # 0 to 511
    
    # Create polygon mask
    # polygon() expects (row, col) = (y, x)
    rr, cc = polygon(y_int, x_int, shape=(height, width))
    mask = np.zeros((height, width), dtype=bool)  # (512, 512)
    mask[rr, cc] = True
    
    # Extract mean response
    time_series = np.zeros(n_frames)  # length = 53
    
    # Loop through frames (frames are first dimension)
    for frame_idx in range(n_frames):
        # Get current frame: shape (512, 512)
        current_frame = dff[frame_idx, :, :]
        
        # Extract pixels within mask and compute mean
        region_pixels = current_frame[mask]
        time_series[frame_idx] = np.mean(region_pixels)
    
    return time_series, mask


def plot_brain_region_temporal_profiles(df_mean_ls, conditions, wf_sf=10, regions_to_plot=None, figsize=(10, 30),fig_file='',title=''):
    
    stim_onset = 1.25 
    stim_duration = 3
    feedback_onset = 2.25
    if regions_to_plot is None:
        all_regions = sorted(list(set(col[:-2] for col in df_mean_ls[0].columns 
                                    if col.endswith(('_l', '_r')))))
        regions_to_plot = all_regions[:]  # First 8 regions
    
    n_regions = len(regions_to_plot)
    
    fig, axes = plt.subplots(n_regions, 4, figsize=figsize, 
                            sharex=False, sharey='row')
    
    if n_regions == 1:
        axes = axes.reshape(1, -1)
        
    max_times = []
    for col_idx, df in enumerate(df_mean_ls):
        max_time = len(df) / wf_sf
        max_times.append(max_time)
        # print(f"Column {col_idx} ({conditions[col_idx]}): max time = {max_time:.2f}s")
        
    hemisphere_colors = {'left': '#1f77b4', 'right': '#ff7f0e'}
    
    for row_idx, region in enumerate(regions_to_plot):
        for col_idx, (df, cond) in enumerate(zip(df_mean_ls, conditions)):
            ax = axes[row_idx, col_idx]
            time_axis = np.arange(len(df)) / wf_sf
            # print(time_axis[-1])
            
            # Plot both hemispheres
            for hemisphere, color in hemisphere_colors.items():
                col_name = f"{region}_{hemisphere[0]}"  # 'l' or 'r'
                if col_name in df.columns:
                    linestyle = '-' if hemisphere == 'left' else '-'
                    ax.plot(time_axis, df[col_name], 
                           color=color, linestyle=linestyle,
                           linewidth=1.5, alpha=0.8,
                           label=hemisphere.capitalize())
                    ax.axvline(x=stim_onset,color='green',linestyle='--')
                    ax.axvline(x=(stim_onset+stim_duration),color='green',linestyle='--')
                    ax.axvline(x=feedback_onset,color='orange',linestyle='--')
            ax.set_xlim(0, max_times[col_idx])
            
            # Set labels
            if row_idx == 0:
                ax.set_title(cond, fontsize=11, fontweight='bold')
            if col_idx == 0:
                ax.set_ylabel(f"{region}", fontsize=10)
            if row_idx == n_regions - 1:
                ax.set_xlabel('Time (s)', fontsize=9)
            else:
                ax.set_xticklabels([])  # Remove x tick labels for non-bottom rows
            
            # ax.grid(True, alpha=0.2)
            
            # Add legend to first subplot
            if row_idx == 0 and col_idx == len(conditions)-1:
                ax.legend(fontsize=8)
    
    plt.suptitle(title,
                 fontsize=12, fontweight='bold', y=1)
    plt.tight_layout()
    plt.subplots_adjust(top=0.98,  # Make room for suptitle
                   bottom=0.05,  # Bottom margin
                   left=0.08,  # Left margin
                   right=0.98,  # Right margin
                   wspace=0.15,  # Horizontal space between subplots
                   hspace=0.25)  # Vertical space between subplots
    if len(fig_file)>0:
        savefig(fig_file)
    plt.show()

def response_latency(signal, baseline_s=1.25, fs=10, thr_std=5):
    """Response onset latency (s) relative to baseline."""
    n_base = int(baseline_s * fs)
    base = signal[:n_base]

    if len(base) == 0 or np.std(base) == 0:
        return np.nan

    thr = np.mean(base) + thr_std * np.std(base)
    idx = np.where(signal[n_base:] > thr)[0]

    return (idx[0] + n_base) / fs if len(idx) else np.nan


def response_amplitude(signal, baseline_s=1.25, fs=10):
    """Response amplitude relative to baseline mean."""
    n_base = int(baseline_s * fs)
    base = signal[:n_base]
    return np.nanmax(signal) - np.nanmean(base)



def compute_pairwise_matrix(df, regions, measure, fs):
    """
    Compute pairwise latency/amplitude differences AND
    return absolute latency/amplitude per region.
    """

    # absolute latency or amplitude per region
    region_values = pd.Series(
        {
            r: (
                response_latency(df[r].values, fs=fs)
                if measure == 'latency'
                else response_amplitude(df[r].values, fs=fs)
            )
            for r in regions
        }
    )

    # pairwise difference matrix (j - i)
    mat = np.full((len(regions), len(regions)), np.nan)
    for i, ri in enumerate(regions):
        for j, rj in enumerate(regions):
            if i != j and np.isfinite(region_values[ri]) and np.isfinite(region_values[rj]):
                mat[i, j] = region_values[rj] - region_values[ri]

    pairwise_df = pd.DataFrame(mat, index=regions, columns=regions)

    return pairwise_df, region_values

def visualize_hemisphere_matrix(mat_df, condition, measure):
    regions = mat_df.index.tolist()
    base_regions = sorted(set(r[:-2] for r in regions))
    unit = 's' if measure == 'latency' else 'Δsignal'
    title_measure = 'Latency' if measure == 'latency' else 'Amplitude'

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    vals = mat_df.values.copy()
    np.fill_diagonal(vals, 0)

    vmax = np.nanmax(np.abs(vals))
    vmin = -vmax if vmax > 0 else -1

    # 1. Full matrix
    im = axes[0, 0].imshow(vals, cmap='coolwarm', vmin=vmin, vmax=vmax)
    axes[0, 0].set_title(f'{condition}\nFull {title_measure} Difference Matrix ({unit})')
    axes[0, 0].set_xticks(range(len(regions)))
    axes[0, 0].set_yticks(range(len(regions)))
    axes[0, 0].set_xticklabels(regions, rotation=45, ha='right', fontsize=8)
    axes[0, 0].set_yticklabels(regions, fontsize=8)
    plt.colorbar(im, ax=axes[0, 0], label=f'{title_measure} difference ({unit})')

    # 2. Inter-hemispheric (L→R)
    diffs, labels = [], []
    for r in base_regions:
        l, rgt = f'{r}_l', f'{r}_r'
        if l in mat_df.index and rgt in mat_df.columns:
            diffs.append(mat_df.loc[l, rgt])
            labels.append(r)

    ax = axes[0, 1]
    colors = ['blue' if d > 0 else 'red' for d in diffs]
    ax.bar(range(len(diffs)), diffs, color=colors)
    ax.axhline(0, color='k', alpha=0.3)
    ax.set_title(f'Inter-hemispheric {title_measure} (R − L)')
    ax.set_ylabel(f'{title_measure} difference ({unit})')
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha='right')

    # 3. Left hemisphere
    left = [r for r in regions if r.endswith('_l')]
    idx = [regions.index(r) for r in left]
    im = axes[1, 0].imshow(vals[np.ix_(idx, idx)], cmap='coolwarm', vmin=vmin, vmax=vmax)
    axes[1, 0].set_title(f'Left Hemisphere {title_measure}')
    axes[1, 0].set_xticks(range(len(left)))
    axes[1, 0].set_yticks(range(len(left)))
    axes[1, 0].set_xticklabels([r[:-2] for r in left], rotation=45, ha='right')
    axes[1, 0].set_yticklabels([r[:-2] for r in left])
    plt.colorbar(im, ax=axes[1, 0], label=f'{title_measure} difference ({unit})')

    # 4. Right hemisphere
    right = [r for r in regions if r.endswith('_r')]
    idx = [regions.index(r) for r in right]
    im = axes[1, 1].imshow(vals[np.ix_(idx, idx)], cmap='coolwarm', vmin=vmin, vmax=vmax)
    axes[1, 1].set_title(f'Right Hemisphere {title_measure}')
    axes[1, 1].set_xticks(range(len(right)))
    axes[1, 1].set_yticks(range(len(right)))
    axes[1, 1].set_xticklabels([r[:-2] for r in right], rotation=45, ha='right')
    axes[1, 1].set_yticklabels([r[:-2] for r in right])
    plt.colorbar(im, ax=axes[1, 1], label=f'{title_measure} difference ({unit})')

    plt.suptitle(f'{title_measure} Analysis: {condition}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

    return fig

def analyze_all_conditions(df_mean_ls, conditions, fs=10, measure='latency'):
    results = {}

    df0 = df_mean_ls[0]
    base = sorted(set(c[:-2] for c in df0.columns if c.endswith(('_l', '_r'))))
    regions = [f'{r}_l' for r in base] + [f'{r}_r' for r in base]

    for df, cond in zip(df_mean_ls, conditions):
        print(f'Analyzing {cond} ({measure})...')
        mat,_ = compute_pairwise_matrix(df, regions, measure, fs)
        results[cond] = mat
        visualize_hemisphere_matrix(mat, cond, measure)

    return results


def granger_causality(x, y, lag=2):
    """
    Safe Granger Causality test with error handling.
    """
    n = min(len(x), len(y))
    
    # Ensure enough data points
    if n < 3 * lag + 10:  # Minimum data points requirement
        return 0, 1.0, False
    
    # Create lagged vectors
    y_data = y[lag:n]
    y_past = np.column_stack([y[lag-i:n-i] for i in range(1, lag+1)])
    
    # Check for NaN or infinite values
    if np.any(np.isnan(y_data)) or np.any(np.isinf(y_data)):
        return 0, 1.0, False
    
    # Add small regularization to avoid singular matrices
    regularization = 1e-8 * np.eye(y_past.shape[1])
    
    try:
        # Models - Restricted: only Y's past
        X1 = np.column_stack([np.ones(len(y_past)), y_past])
        
        # Unrestricted: Y's past + X's past
        x_past = np.column_stack([x[lag-i:n-i] for i in range(1, lag+1)])
        X2 = np.column_stack([np.ones(len(y_past)), y_past, x_past])
        
        # Add regularization to design matrices
        X1_reg = X1.copy()
        X2_reg = X2.copy()
        
        # Fit models with error handling
        try:
            beta1 = np.linalg.lstsq(X1_reg, y_data, rcond=None)[0]
            beta2 = np.linalg.lstsq(X2_reg, y_data, rcond=None)[0]
        except np.linalg.LinAlgError:
            # Use pseudo-inverse as fallback
            beta1 = np.linalg.pinv(X1_reg) @ y_data
            beta2 = np.linalg.pinv(X2_reg) @ y_data
        
        # Calculate residuals
        res1 = y_data - X1_reg @ beta1
        res2 = y_data - X2_reg @ beta2
        
        # RSS
        rss1 = np.sum(res1**2)
        rss2 = np.sum(res2**2)
        
        # Avoid division by zero
        if rss2 < 1e-10:
            return 0, 1.0, False
            
        # F-test
        f_stat = ((rss1 - rss2) / lag) / (rss2 / (len(y_data) - 2*lag - 1))
        
        # Calculate proper p-value
        df_num = lag
        df_den = len(y_data) - 2*lag - 1
        
        if df_den > 0 and f_stat > 0:
            p_value = 1 - f.cdf(f_stat, df_num, df_den)
        else:
            p_value = 1.0
            
        return f_stat, p_value, rss1 > rss2
        
    except Exception as e:
        # Catch any other errors
        warnings.warn(f"Granger causality calculation failed: {e}")
        return 0, 1.0, False


def transfer_entropy(x, y, lag=1):
    """
    Safe Transfer Entropy calculation with error handling.
    """
    try:
        n = min(len(x), len(y)) - lag
        
        if n < 10:  # Minimum data points
            return 0.0
            
        # Align data
        x_past = x[:n]
        y_past = y[lag:lag+n]
        y_future = y[lag:lag+n]
        
        # Use bins based on quantiles
        x_bin = (x_past > np.median(x_past)).astype(int)
        y_past_bin = (y_past > np.median(y_past)).astype(int)
        y_future_bin = (y_future > np.median(y_future)).astype(int)
        
        # Count states with smoothing
        count = np.ones((2, 2, 2))
        
        for i in range(n):
            count[y_future_bin[i], y_past_bin[i], x_bin[i]] += 1
        
        # Normalize
        p = count / (n + 8)
        
        # Calculate TE
        te = 0
        for yf in [0, 1]:
            for yp in [0, 1]:
                for xp in [0, 1]:
                    p_joint = p[yf, yp, xp]
                    p_yx = np.sum(p[:, yp, xp])
                    p_yy = np.sum(p[yf, yp, :])
                    p_y = np.sum(p[:, yp, :])
                    
                    if p_yx > 0 and p_y > 0 and p_joint > 0:
                        p_cond1 = p_joint / p_yx
                        p_cond2 = p_yy / p_y
                        
                        if p_cond1 > 0 and p_cond2 > 0:
                            te += p_joint * np.log2(p_cond1 / p_cond2)
        
        return max(0, te)
        
    except Exception as e:
        warnings.warn(f"Transfer entropy calculation failed: {e}")
        return 0.0


def find_optimal_lag(x, y, max_lag=10, method='granger'):
    """
    Find optimal lag for causality analysis.
    
    Parameters:
    -----------
    x, y : array-like
        Time series data
    max_lag : int
        Maximum lag to test
    method : str
        'granger' or 'transfer'
    
    Returns:
    --------
    optimal_lag : int
        Optimal lag value
    scores : list
        Scores for each lag
    """
    scores = []
    valid_lags = []
    
    for lag in range(1, max_lag + 1):
        try:
            if method == 'granger':
                f_stat, p_value, _ = granger_causality(x, y, lag=lag)
                # Use -log10(p) as score, lower p = better
                if p_value > 0:
                    score = -np.log10(p_value)
                else:
                    score = 0
            else:  # transfer entropy
                te = transfer_entropy(x, y, lag=lag)
                score = te
                
            scores.append(score)
            valid_lags.append(lag)
            
        except Exception:
            scores.append(0)
            valid_lags.append(lag)
    
    if len(scores) == 0:
        return 1, [0] * max_lag
    
    # Find lag with maximum score
    optimal_idx = np.argmax(scores)
    optimal_lag = valid_lags[optimal_idx]
    
    return optimal_lag, scores


def plot_causality_matrix(df, regions, causality_type='granger', lag='auto', figsize=(12, 10),title=''):
    """
    Plot Granger Causality or Transfer Entropy matrix for brain regions.
    
    Parameters:
    -----------
    df : DataFrame
        Time series data with regions as columns
    regions : list
        List of region names (e.g., ['VISp_l', 'VISp_r', ...])
    causality_type : str
        'granger' for Granger Causality or 'transfer' for Transfer Entropy
    lag : int or 'auto'
        Lag parameter for causality calculation, 'auto' for automatic selection
    figsize : tuple
        Figure size
    
    Returns:
    --------
    fig : matplotlib figure
        Figure object
    causality_matrix : ndarray
        Causality matrix
    p_value_matrix : ndarray
        P-value matrix (for Granger only)
    optimal_lags : ndarray
        Optimal lags used for each pair (if lag='auto')
    """
    
    n_regions = len(regions)
    causality_matrix = np.zeros((n_regions, n_regions))
    p_value_matrix = np.ones((n_regions, n_regions))
    optimal_lags = np.zeros((n_regions, n_regions), dtype=int)
    
    # Determine lag strategy
    if lag == 'auto':
        # Use optimal lag for each pair
        print("Finding optimal lags for each pair...")
        for i in tqdm(range(n_regions), desc="Finding optimal lags"):
            for j in range(n_regions):
                if i != j:
                    x = df[regions[i]].values
                    y = df[regions[j]].values
                    optimal_lag, _ = find_optimal_lag(x, y, method=causality_type)
                    optimal_lags[i, j] = optimal_lag
    else:
        # Use fixed lag for all pairs
        optimal_lags[:, :] = lag
    
    # Calculate causality for each pair
    for i in tqdm(range(n_regions), desc=f"Calculating {causality_type}"):
        for j in range(n_regions):
            if i != j:
                x = df[regions[i]].values
                y = df[regions[j]].values
                current_lag = optimal_lags[i, j]
                
                if causality_type == 'granger':
                    f_stat, p_value, _ = granger_causality(x, y, lag=current_lag)
                    causality_matrix[i, j] = f_stat
                    p_value_matrix[i, j] = p_value
                elif causality_type == 'transfer':
                    te = transfer_entropy(x, y, lag=current_lag)
                    causality_matrix[i, j] = te
                else:
                    raise ValueError("causality_type must be 'granger' or 'transfer'")
    
    # Create figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Full causality matrix
    vmax = np.nanmax(causality_matrix[causality_matrix > 0]) if np.any(causality_matrix > 0) else 1
    im1 = axes[0, 0].imshow(causality_matrix, cmap='hot', vmin=0, vmax=vmax)
    axes[0, 0].set_title(f'{causality_type.capitalize()} Matrix\n(Full Brain)')
    axes[0, 0].set_xticks(range(n_regions))
    axes[0, 0].set_yticks(range(n_regions))
    axes[0, 0].set_xticklabels(regions, rotation=45, ha='right', fontsize=8)
    axes[0, 0].set_yticklabels(regions, fontsize=8)
    plt.colorbar(im1, ax=axes[0, 0], label=f'{causality_type.capitalize()} Value')
    
    # Add significance stars for Granger causality
    if causality_type == 'granger':
        sig_threshold = 0.05
        for i in range(n_regions):
            for j in range(n_regions):
                if i != j and p_value_matrix[i, j] < sig_threshold:
                    axes[0, 0].text(j, i, '*', ha='center', va='center', 
                                   color='white', fontsize=8, fontweight='bold')
    
    # 2. Lag matrix (if auto lag was used)
    im2 = axes[0, 1].imshow(optimal_lags, cmap='viridis', vmin=1, vmax=10)
    axes[0, 1].set_title('Optimal Lags Used')
    axes[0, 1].set_xticks(range(n_regions))
    axes[0, 1].set_yticks(range(n_regions))
    axes[0, 1].set_xticklabels(regions, rotation=45, ha='right', fontsize=8)
    axes[0, 1].set_yticklabels(regions, fontsize=8)
    plt.colorbar(im2, ax=axes[0, 1], label='Lag (frames)')
    
    # 3. Top connections
    axes[0, 2].axis('off')
    if causality_type == 'granger':
        text_content = f"Granger Causality Analysis\n"
        text_content += f"Significance: * p < 0.05\n\n"
        if lag != 'auto':
            text_content += f"Fixed lag: {lag}\n\n"
    else:
        text_content = f"Transfer Entropy Analysis\n"
        if lag != 'auto':
            text_content += f"Fixed lag: {lag}\n\n"
    
    # Find top connections
    flat_indices = np.argsort(causality_matrix.flatten())[::-1]
    top_n = min(8, n_regions * n_regions - n_regions)  # Exclude diagonal
    
    text_content += "Top Connections:\n"
    count = 0
    for idx in flat_indices:
        if count >= top_n:
            break
        i, j = divmod(idx, n_regions)
        if i != j and causality_matrix[i, j] > 0:
            if causality_type == 'granger':
                sig_star = '*' if p_value_matrix[i, j] < 0.05 else ''
                lag_info = f" (lag={optimal_lags[i, j]})" if lag == 'auto' else ""
                text_content += f"{regions[i]} → {regions[j]}: {causality_matrix[i, j]:.3f}{sig_star}{lag_info}\n"
            else:
                lag_info = f" (lag={optimal_lags[i, j]})" if lag == 'auto' else ""
                text_content += f"{regions[i]} → {regions[j]}: {causality_matrix[i, j]:.3f}{lag_info}\n"
            count += 1
    
    if count == 0:
        text_content += "No significant connections found\n"
    
    axes[0, 2].text(0.1, 0.5, text_content, transform=axes[0, 2].transAxes,
                   verticalalignment='center', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 4. Left hemisphere only
    left_regions = [r for r in regions if r.endswith('_l')]
    if len(left_regions) > 0:
        left_indices = [regions.index(r) for r in left_regions]
        left_matrix = causality_matrix[np.ix_(left_indices, left_indices)]
        
        im2 = axes[1, 0].imshow(left_matrix, cmap='hot', vmin=0, vmax=vmax)
        axes[1, 0].set_title(f'{causality_type.capitalize()} Matrix\n(Left Hemisphere)')
        axes[1, 0].set_xticks(range(len(left_regions)))
        axes[1, 0].set_yticks(range(len(left_regions)))
        axes[1, 0].set_xticklabels([r[:-2] for r in left_regions], rotation=45, ha='right')
        axes[1, 0].set_yticklabels([r[:-2] for r in left_regions])
        plt.colorbar(im2, ax=axes[1, 0], label=f'{causality_type.capitalize()} Value')
        
        if causality_type == 'granger':
            left_p_matrix = p_value_matrix[np.ix_(left_indices, left_indices)]
            for i in range(len(left_regions)):
                for j in range(len(left_regions)):
                    if i != j and left_p_matrix[i, j] < 0.05:
                        axes[1, 0].text(j, i, '*', ha='center', va='center', 
                                       color='white', fontsize=8, fontweight='bold')
    
    # 5. Right hemisphere only
    right_regions = [r for r in regions if r.endswith('_r')]
    if len(right_regions) > 0:
        right_indices = [regions.index(r) for r in right_regions]
        right_matrix = causality_matrix[np.ix_(right_indices, right_indices)]
        
        im3 = axes[1, 1].imshow(right_matrix, cmap='hot', vmin=0, vmax=vmax)
        axes[1, 1].set_title(f'{causality_type.capitalize()} Matrix\n(Right Hemisphere)')
        axes[1, 1].set_xticks(range(len(right_regions)))
        axes[1, 1].set_yticks(range(len(right_regions)))
        axes[1, 1].set_xticklabels([r[:-2] for r in right_regions], rotation=45, ha='right')
        axes[1, 1].set_yticklabels([r[:-2] for r in right_regions])
        plt.colorbar(im3, ax=axes[1, 1], label=f'{causality_type.capitalize()} Value')
        
        if causality_type == 'granger':
            right_p_matrix = p_value_matrix[np.ix_(right_indices, right_indices)]
            for i in range(len(right_regions)):
                for j in range(len(right_regions)):
                    if i != j and right_p_matrix[i, j] < 0.05:
                        axes[1, 1].text(j, i, '*', ha='center', va='center', 
                                       color='white', fontsize=8, fontweight='bold')
    
    # 6. Lag distribution (if auto lag was used)
    if lag == 'auto':
        axes[1, 2].hist(optimal_lags[optimal_lags > 0].flatten(), 
                       bins=range(1, 12), edgecolor='black', alpha=0.7)
        axes[1, 2].set_title('Optimal Lag Distribution')
        axes[1, 2].set_xlabel('Lag (frames)')
        axes[1, 2].set_ylabel('Frequency')
        axes[1, 2].set_xticks(range(1, 11))
    else:
        axes[1, 2].axis('off')
    
    # # Adjust layout
    # title = f'{causality_type.upper()} Analysis'
    # if lag == 'auto':
    #     title += ' (Auto Lag)'
    # else:
    #     title += f' (Lag={lag})'
    
    plt.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if causality_type == 'granger':
        return fig, causality_matrix, p_value_matrix, optimal_lags
    else:
        return fig, causality_matrix, None, optimal_lags


def analyze_all_causality(df_mean_ls, conditions, causality_type='granger', lag='auto'):
    """
    Analyze causality for all conditions.
    
    Parameters:
    -----------
    df_mean_ls : list of DataFrames
        List of dataframes for each condition
    conditions : list
        List of condition names
    causality_type : str
        'granger' or 'transfer'
    lag : int or 'auto'
        Lag parameter or 'auto' for automatic selection
    
    Returns:
    --------
    results : dict
        Dictionary of causality matrices for each condition
    """
    
    results = {}
    p_values = {}
    optimal_lags_all = {}
    
    # Get regions from first dataframe
    df0 = df_mean_ls[0]
    base_regions = sorted(set(c[:-2] for c in df0.columns if c.endswith(('_l', '_r'))))
    regions = [f'{r}_l' for r in base_regions] + [f'{r}_r' for r in base_regions]
    
    # Filter regions that exist in all dataframes
    common_regions = []
    for r in regions:
        if all(r in df.columns for df in df_mean_ls):
            common_regions.append(r)
    
    print(f"Analyzing {causality_type} causality for {len(common_regions)} regions")
    print(f"Regions: {common_regions}")
    
    # Analyze each condition
    for df, cond in zip(df_mean_ls, conditions):
        print(f'\n{"="*50}')
        print(f'Condition: {cond}')
        print(f'{"="*50}')
        
        fig, causality_matrix, p_value_matrix, optimal_lags = plot_causality_matrix(
            df, common_regions, causality_type=causality_type, lag=lag, title=cond
        )
        
        results[cond] = causality_matrix
        if p_value_matrix is not None:
            p_values[cond] = p_value_matrix
        optimal_lags_all[cond] = optimal_lags
        
        # # Save figure
        # lag_str = 'auto' if lag == 'auto' else f'lag{lag}'
        # fig_file = f'{causality_type}_{lag_str}_{cond}.png'
        # plt.savefig(fig_file, dpi=300, bbox_inches='tight')
        # print(f"Saved figure: {fig_file}")
        # plt.show()
    
    return results, p_values, optimal_lags_all


def list_folders_starting_with_202_glob(path):
    """List all folders starting with '2025' using glob pattern matching."""
    # Join path with pattern (ensure path ends with separator)
    if not path.endswith(os.sep):
        path = path + os.sep
    
    # Pattern to match folders starting with 2025
    pattern = os.path.join(path, "202*")
    
    # Use glob to find matching directories
    matching_dirs = []
    for item in glob(pattern):
        if os.path.isdir(item):
            # Get just the folder name
            folder_name = os.path.basename(item)
            matching_dirs.append(folder_name)
    
    return matching_dirs


def plot_region_temporal_profiles(df_mean_ls_all_dates, dates_list, region_name, conditions, 
                                  wf_sf=10, figsize=(16, 12), fig_file='', title=''):
    """
    Plot temporal profiles for a specific brain region across dates and conditions.
    
    Parameters:
    -----------
    df_mean_ls_all_dates : list of lists
        List where each element corresponds to a date and contains a list of 
        DataFrames for each condition [date][condition]
    dates_list : list
        List of date strings (e.g., ['20250827', '20250828', ...])
    region_name : str
        Name of brain region to plot (e.g., 'VISp')
    conditions : list
        List of condition names (e.g., ['Hit', 'FA', 'CR', 'Miss'])
    wf_sf : float
        Widefield sampling frequency (Hz)
    figsize : tuple
        Figure size
    fig_file : str
        Path to save figure
    title : str
        Figure title
    
    Returns:
    --------
    fig : matplotlib Figure object
    """
    
    n_dates = len(dates_list)
    n_conditions = len(conditions)
    
    # Create figure with subplots: n_dates rows, n_conditions columns
    fig, axes = plt.subplots(n_dates, n_conditions, figsize=figsize, 
                            sharex=True, sharey='row', constrained_layout=True)
    
    # Handle case when there's only one date
    if n_dates == 1:
        axes = axes.reshape(1, -1)
    
    # Convert dates to readable format
    date_objects = pd.to_datetime(dates_list, format='%Y%m%d')
    dates_mmdd = [d.strftime('%m/%d') for d in date_objects]
    
    # Define stim timings
    stim_onset = 1.25
    stim_duration = 3
    feedback_onset = 2.25
    
    # Define colors for hemispheres
    hemisphere_colors = {'left': '#1f77b4', 'right': '#ff7f0e'}
    
    # Define colors for conditions (optional, can be used for titles)
    condition_colors = {
        'Hit': 'green',
        'FA': 'red', 
        'CR': 'blue',
        'Miss': 'orange'
    }
    
    # Find global y-axis limits for consistent scaling across dates
    global_y_min = np.inf
    global_y_max = -np.inf
    
    for date_idx in range(n_dates):
        for cond_idx in range(n_conditions):
            df = df_mean_ls_all_dates[date_idx][cond_idx]
            
            # Check if region exists in this dataset
            left_col = f"{region_name}_l"
            right_col = f"{region_name}_r"
            
            if left_col in df.columns:
                y_vals = df[left_col].values
                global_y_min = min(global_y_min, np.nanmin(y_vals))
                global_y_max = max(global_y_max, np.nanmax(y_vals))
            
            if right_col in df.columns:
                y_vals = df[right_col].values
                global_y_min = min(global_y_min, np.nanmin(y_vals))
                global_y_max = max(global_y_max, np.nanmax(y_vals))
    
    # Add some padding to y-limits
    y_padding = 0.1 * (global_y_max - global_y_min)
    global_y_min = global_y_min - y_padding
    global_y_max = global_y_max + y_padding
    
    # Plot data
    for date_idx in range(n_dates):
        for cond_idx in range(n_conditions):
            ax = axes[date_idx, cond_idx]
            df = df_mean_ls_all_dates[date_idx][cond_idx]
            
            # Create time axis
            time_axis = np.arange(len(df)) / wf_sf
            
            # Check if region exists in this dataset
            left_col = f"{region_name}_l"
            right_col = f"{region_name}_r"
            
            has_left = left_col in df.columns
            has_right = right_col in df.columns
            
            # Plot both hemispheres if available
            if has_left:
                ax.plot(time_axis, df[left_col], 
                       color=hemisphere_colors['left'], 
                       linewidth=1.5, alpha=0.8,
                       label='Left' if date_idx == 0 and cond_idx == 0 else None)
            
            if has_right:
                ax.plot(time_axis, df[right_col], 
                       color=hemisphere_colors['right'], 
                       linewidth=1.5, alpha=0.8,
                       label='Right' if date_idx == 0 and cond_idx == 0 else None)
            
            # Add stimulus and feedback markers
            ax.axvline(x=stim_onset, color='green', linestyle='--', alpha=0.6, linewidth=0.8)
            ax.axvline(x=(stim_onset + stim_duration), color='green', linestyle='--', alpha=0.6, linewidth=0.8)
            ax.axvline(x=feedback_onset, color='orange', linestyle='--', alpha=0.6, linewidth=0.8)
            
            # Fill stimulus period
            ax.axvspan(stim_onset, stim_onset + stim_duration, 
                      alpha=0.1, color='green', label='Stimulus' if date_idx == 0 and cond_idx == 0 else None)
            
            # Set axis limits
            ax.set_xlim(0, time_axis[-1])
            ax.set_ylim(global_y_min, global_y_max)
            
            # Set labels and titles
            if date_idx == 0:  # Top row: condition titles
                ax.set_title(conditions[cond_idx], fontsize=10, fontweight='bold',
                            color=condition_colors.get(conditions[cond_idx], 'black'))
            
            if cond_idx == 0:  # Left column: date labels
                ax.set_ylabel(f"{dates_mmdd[date_idx]}\n{region_name}", fontsize=9)
            
            if date_idx == n_dates - 1:  # Bottom row: x-axis labels
                ax.set_xlabel('Time (s)', fontsize=8)
            
            # Add grid
            ax.grid(True, alpha=0.1, linestyle='--')
            
            # Set tick parameters
            ax.tick_params(axis='both', labelsize=7)
    
    # Add legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        # Create custom legend entries for stimulus period
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        
        custom_handles = []
        custom_labels = []
        
        # Add hemisphere lines
        if has_left or has_right:
            custom_handles.append(Line2D([0], [0], color=hemisphere_colors['left'], linewidth=1.5))
            custom_labels.append('Left')
            custom_handles.append(Line2D([0], [0], color=hemisphere_colors['right'], linewidth=1.5))
            custom_labels.append('Right')
        
        # Add stimulus markers
        custom_handles.append(Patch(facecolor='green', alpha=0.1, edgecolor='green'))
        custom_labels.append('Stimulus')
        custom_handles.append(Line2D([0], [0], color='green', linestyle='--', linewidth=0.8))
        custom_labels.append('Stim On/Off')
        custom_handles.append(Line2D([0], [0], color='orange', linestyle='--', linewidth=0.8))
        custom_labels.append('Feedback')
        
        # Add legend in the last subplot
        axes[-1, -1].legend(custom_handles, custom_labels, 
                           fontsize=7, loc='upper right',
                           framealpha=0.8, frameon=True)
    
    # Add overall title
    fig.suptitle(f"{title}\n{region_name} Temporal Profiles", 
                 fontsize=12, fontweight='bold', y=1.02)
    
    # Save figure if requested
    if fig_file:
        os.makedirs(os.path.dirname(fig_file), exist_ok=True)
        plt.savefig(fig_file, dpi=300, bbox_inches='tight')
        print(f"Figure saved to: {fig_file}")
    
    plt.show()
    return


# Example usage with your existing data structure:
# First, you need to reorganize your data to match the function's expected input format

# Assuming you have a structure like this in your main loop:
# df_mean_ls_all_dates = []  # Initialize outside the loop

# Inside your date loop:
# df_mean_ls = []  # List for current date
# for tp in type_list:
#     # Process each trial type and append to df_mean_ls
#     df_mean_ls.append(df_mean)
# df_mean_ls_all_dates.append(df_mean_ls)  # Append list for this date

# After collecting all dates, call:
# plot_region_temporal_profiles(
#     df_mean_ls_all_dates=df_mean_ls_all_dates,
#     dates_list=dates_ls,
#     region_name='VISp',
#     conditions=type_list,
#     wf_sf=10,
#     figsize=(16, 4 * len(dates_ls)),  # Adjust height based on number of dates
#     fig_file=os.path.join(base_dir, mouse_id, f'{mouse_id}_VISp_temporal_profiles.png'),
#     title=f'Mouse {mouse_id}'
# )
