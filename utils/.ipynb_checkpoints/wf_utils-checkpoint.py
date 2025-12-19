from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import os
import scipy
import yaml
from PIL import Image, ImageDraw, ImageFont

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
        if len(_path_470) == 1:
            image_470 = imread(_path_470[0])
            image_470 = image_470 * mask if mask is not None else image_470
            if crop_params is not None:
                image_470 = rotate_crop_array(image_470, angle=angle,
                    left=left, top=top, width=width, height=height)
            all_images[i, 0, :, :] = image_470
        else:
            print(f'File {i}.tif not found in 470 channel!')

        if len(_path_405) == 1:
            image_405 = imread(_path_405[0])
            image_405 = image_405 * mask if mask is not None else image_405
            if crop_params is not None:
                image_405 = rotate_crop_array(image_405, angle=angle,
                    left=left, top=top, width=width, height=height)
            all_images[i, 1, :, :] = image_405
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

def preprocess_mice(config_path):
    config = load_config(config_path)
    for session in config["sessions"]:
        load_timelite(config, session)
    print(f"all done")
    
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
    
    for value in y:
        # Calculate absolute differences
        differences = np.abs(x - value)
        # Find index of minimum difference
        closest_idx = np.argmin(differences)
        indices.append(closest_idx)
    
    return np.array(indices)


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


def compute_trial_mean(dff, idx_onset, idx_offset):
    """
    Vectorized version using padding for speed.
    """
    
    # Calculate trial lengths
    trial_lengths = idx_offset - idx_onset
    
    # Find the two possible lengths
    unique_lengths = np.unique(trial_lengths)
    
    # Choose the shorter length
    n_frames= np.min(unique_lengths)
    print(n_frames)
    n_trials = len(idx_onset)

    _, x, y = dff.shape
    
    # Pre-allocate array for all trials
    # Shape: [n_trials, n_frames, x, y]
    all_trials = np.full((n_trials, n_frames, x, y), np.nan, dtype=np.float32)
    
    # Extract and pad each trial
    for i in tqdm(range(n_trials)):
        start = idx_onset[i]
        end = start+n_frames
        print(start)
        trial_data = dff[start:end] 
        trial_len = trial_data.shape[0]
        # Pad to max length
        all_trials[i, :trial_len] = trial_data
    # Compute nanmean across trials
    dff_mean = np.nanmean(all_trials, axis=0)
    return dff_mean
