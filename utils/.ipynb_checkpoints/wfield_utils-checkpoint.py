from wfield import *
from tifffile import imwrite
import pandas as pd


'''
def reconstruct(u,svt,dims = None):
    if issparse(u):
        if dims is None:
            raise ValueError('Supply dims = [H,W] when using sparse arrays')
    else:
        if dims is None:
            dims = u.shape[:2]
    return u.dot(svt).reshape((*dims,-1)).transpose(-1,0,1).squeeze()
'''
#%%
def svd2tif(path_wfield, name='', uint16=False, corr470=False):
    # 读取SVD文件
    U = np.load(pjoin(path_wfield, 'U.npy'))
    SVTcorr = np.load(pjoin(path_wfield, 'SVTcorr.npy'))
    frames_average = np.load(pjoin(path_wfield, 'frames_average.npy'))

    # 重构矫正后图像
    SVD_corr = reconstruct(U, SVTcorr)
    if uint16 is False:
        imwrite(pjoin(path_wfield, name + "SVD_corr.tif"), SVD_corr, imagej=True)
        print(name+'SVD_corr.tif输出完成')
    elif uint16 is True:
        # 矫正后图像转成uint16格式
        SVD_corr_uint16 = ((SVD_corr+1)*32768).astype('uint16')
        imwrite(pjoin(path_wfield, name + "SVD_corr_uint16.tif"), SVD_corr_uint16, imagej=True)
        print(name+'SVD_corr_uint16.tif输出完成')
    else:
        print('格式不支持，不输出SVD_corr')

    if corr470 is True:
        # 输出矫正后470通道图像
        tif470_corr = (SVD_corr * frames_average[0, :, :]) + frames_average[0, :, :]
        imwrite(pjoin(path_wfield, name + "hemo-corr.tif"), tif470_corr.astype('uint16'), dtype='uint16', imagej=True)
        print(name+'hemo-corr.tif"输出完成')


#%%
def cal_captured_var(s):
    total_variance = np.sum(s ** 2)  # 计算总方差
    captured_var = np.cumsum(s ** 2) / total_variance * 100  # 捕获的方差百分比
    return captured_var

def plot_captured_var(path_wfield):
    s = np.load(pjoin(path_wfield, 's.npy'))
    captured_variance = cal_captured_var(s)
    # 将捕获的方差绘制成图表
    plt.plot(range(1, len(s) + 1), captured_variance, marker='o', markersize=1, linestyle='-', linewidth=0.5)
    plt.title('Captured Variance vs. Number of Components (k)')
    plt.xlabel('Number of Components (k)')
    plt.ylabel('Captured Variance (%)')
    plt.grid(True)
    plt.savefig(pjoin(path_wfield, 'svd_var.png'), bbox_inches='tight', transparent=False, facecolor='white')
    plt.show()


#%%
def cal_snr(x, axis1, axis2):
    """
    axis1:重复，axis2:时间。重复axis要在时间axis之后！
    """
    snr = np.nanvar(np.nanmean(x, axis=axis1), axis=axis2) / np.nanmean(np.nanvar(x, axis=axis1), axis=axis2)
    return snr


#%%
def phasemap(path_wfield, nrepeats=10, post_trial=3, export_ave_tif=True, export_raw_tif=False,
             plot_snr=True, plot_phasemasp=True, export_phase=True):
    ### 路径和刺激重复次数。（本代码对应的是每个方向连续重复10次再下一个方向）
    path_wfield = path_wfield
    nrepeats = nrepeats
    post_trial = post_trial  # 刺激消失后多取几秒
    path_out = pjoin(path_wfield, '..', os.path.basename(path_wfield)[:16] + 'retinotopy')
    os.makedirs(path_out, exist_ok=True)
    os.makedirs(pjoin(path_out,'tif'), exist_ok=True)

    ### load data
    trials = pd.read_csv(pjoin(path_wfield, "trials.csv"), header=None, dtype=int).values
    nframes_el = min(trials[:20, 3]) + post_trial * 10
    nframes_az = min(trials[20:40, 3]) + post_trial * 10
    U = np.load(pjoin(path_wfield, 'U.npy'))
    SVTcorr = np.load(pjoin(path_wfield, 'SVTcorr.npy'))
    nSVD = SVTcorr.shape[0]

    ### extract trial-data for 4 direction stimuli respectively.
    def _sort_frames(nframes, *ntrials):
        stack = np.empty((nSVD, nframes, 0))
        raw = np.empty((nSVD, 0))
        for i in ntrials:
            stack = np.concatenate((stack, SVTcorr[:, trials[i, 1]:trials[i, 1] + nframes].reshape(nSVD, nframes, 1)),
                                   axis=2)
            raw = np.concatenate((raw, SVTcorr[:, trials[i, 1]:trials[i, 1] + nframes]), axis=1)
        avg = np.mean(stack, axis=2)
        return avg, raw, stack

    avg_up, raw_up, stack_up = _sort_frames(nframes_el, *range(nrepeats * 0, nrepeats * 1))
    avg_down, raw_down, stack_down = _sort_frames(nframes_el, *range(nrepeats * 1, nrepeats * 2))
    avg_left, raw_left, stack_left = _sort_frames(nframes_az, *range(nrepeats * 2, nrepeats * 3))
    avg_right, raw_right, stack_right = _sort_frames(nframes_az, *range(nrepeats * 3, nrepeats * 4))

    if export_ave_tif is True:
        # export trial-average tif
        imwrite(pjoin(path_out, 'tif', 'avg_up.tif'), reconstruct(U, avg_up).astype('float32'), imagej=True)
        imwrite(pjoin(path_out, 'tif', 'avg_down.tif'), reconstruct(U, avg_down).astype('float32'), imagej=True)
        imwrite(pjoin(path_out, 'tif', 'avg_left.tif'), reconstruct(U, avg_left).astype('float32'), imagej=True)
        imwrite(pjoin(path_out, 'tif', 'avg_right.tif'), reconstruct(U, avg_right).astype('float32'), imagej=True)
        print('Finish exporting average tif')

    if export_raw_tif is True:
        imwrite(pjoin(path_out, 'tif', 'rep_up.tif'), reconstruct(U, raw_up).astype('float32'), imagej=True)
        imwrite(pjoin(path_out, 'tif', 'rep_down.tif'), reconstruct(U, raw_down).astype('float32'), imagej=True)
        imwrite(pjoin(path_out, 'tif', 'rep_left.tif'), reconstruct(U, raw_left).astype('float32'), imagej=True)
        imwrite(pjoin(path_out, 'tif', 'rep_right.tif'), reconstruct(U, raw_right).astype('float32'), imagej=True)
        print('Finish exporting rep tif')

    if plot_snr is True:
        snr_up = cal_snr(np.tensordot(U, stack_up, axes=(2, 0)), 3, 2)
        snr_down = cal_snr(np.tensordot(U, stack_down, axes=(2, 0)), 3, 2)
        snr_left = cal_snr(np.tensordot(U, stack_left, axes=(2, 0)), 3, 2)
        snr_right = cal_snr(np.tensordot(U, stack_right, axes=(2, 0)), 3, 2)

        vmin = 0
        vmax = max(snr_up.max(), snr_down.max(), snr_left.max(), snr_right.max())
        fig = plt.figure(figsize=[20, 5])

        axis_up = fig.add_subplot(1, 4, 1)
        plt.imshow(snr_up, vmin=vmin, vmax=vmax)
        plt.axis('off')
        plt.colorbar()
        axis_up.set_title('up_snr')

        axis_down = fig.add_subplot(1, 4, 2)
        plt.imshow(snr_down, vmin=vmin, vmax=vmax)
        plt.axis('off')
        plt.colorbar()
        axis_down.set_title('down_snr')

        axis_left = fig.add_subplot(1, 4, 3)
        plt.imshow(snr_left, vmin=vmin, vmax=vmax)
        plt.axis('off')
        plt.colorbar()
        axis_left.set_title('left_snr')

        axis_right = fig.add_subplot(1, 4, 4)
        plt.imshow(snr_right, vmin=vmin, vmax=vmax)
        plt.axis('off')
        plt.colorbar()
        axis_right.set_title('right_snr')

        plt.suptitle(os.path.basename(path_wfield)[:15]+' retinotopy_snr')
        fig.set_facecolor('white')
        plt.savefig(pjoin(path_out, 'retinotopy_snr.png'), bbox_inches='tight')
        plt.show()

    if plot_phasemasp is True:
        ### computes fft in SVD space
        from scipy.ndimage import gaussian_filter,median_filter
        # mov = runpar(median_filter, U.transpose((2, 0, 1)), size=1)
        # U = np.stack(mov).transpose((1, 2, 0)).astype(np.float32)
        up = reconstruct(U, fft(raw_up.T, axis=0)[nrepeats])
        down = reconstruct(U, fft(raw_down.T, axis=0)[nrepeats])
        left = reconstruct(U, fft(raw_left.T, axis=0)[nrepeats])
        right = reconstruct(U, fft(raw_right.T, axis=0)[nrepeats])
        phase_el = -1. * (np.angle(up) - np.angle(down)) % (2 * np.pi)
        mag_el = (np.abs(up + down) * 2.)
        phase_az = -1. * (np.angle(left) - np.angle(right)) % (2 * np.pi)
        mag_az = (np.abs(left + right) * 2.)

        fig = plt.figure(figsize=[10, 5])
        axis_az = fig.add_subplot(1, 2, 1)
        plt.imshow(im_fftphase_hsv([mag_az, phase_az]))
        plt.axis('off')
        axis_az.set_title('azimuth')
        axis_el = fig.add_subplot(1, 2, 2)
        plt.imshow(im_fftphase_hsv([mag_el, phase_el]))
        plt.axis('off')
        axis_el.set_title('elevation')
        plt.suptitle(os.path.basename(path_wfield)[:15] + ' phasemap')
        fig.set_facecolor('white')
        plt.savefig(pjoin(path_out, 'phasemap_unfiltered.png'), bbox_inches='tight')
        plt.show()

        if export_phase is True:
            ### export phase and magnitude
            np.save(pjoin(path_out, 'phase_el.npy'), phase_el)
            np.save(pjoin(path_out, 'phase_az.npy'), phase_az)
            np.save(pjoin(path_out, 'mag_el.npy'), mag_el)
            np.save(pjoin(path_out, 'mag_az.npy'), mag_az)

        ### plot sign maps
        fig = plt.figure()
        plt.imshow(median_filter(visual_sign_map(phase_az, phase_el), 20),
                   cmap='RdBu_r', clim=[-1, 1])
        plt.colorbar(shrink=0.5)
        plt.axis('off')
        plt.suptitle(os.path.basename(path_wfield)[:15] +' sign_map')
        fig.set_facecolor('white')
        # plt.annotate('window: stim_length + {} s'.format(post_trial), xy=(0, 1), xycoords='axes fraction',
        #                 fontsize=12, rotation=0, va='center')
        plt.savefig(pjoin(path_out, 'signmap.png'), bbox_inches='tight')
        plt.show()


# %%
def tif2mp4(tiff, mp4, clip=0.03, fps=10):
    import numpy as np
    import cv2
    import tifffile

    image = tifffile.imread(tiff)
    # Threshold the values
    image_clipped = np.clip(image, -clip, clip)
    # Convert the image to UInt8 format
    uint8_image = ((image_clipped + clip) / clip * 127).astype(np.uint8)
    # Create a VideoWriter object to save the video as MP4
    output_video = cv2.VideoWriter(mp4, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                   (uint8_image.shape[2], uint8_image.shape[1]))
    # Write the images to the video
    for frame in uint8_image:
        output_video.write(frame)
    # Release the VideoWriter object
    output_video.release()

import matplotlib.pyplot as plt
import numpy as np

def filename2int(x):
    import os
    # 从文件名中提取整数值，用于后续排序等。
    filename = os.path.splitext(os.path.basename(x))[0]
    return int(filename)


def log_progress(sequence, every=None, size=None, name='Items'):
    from ipywidgets import IntProgress, HTML, VBox
    from IPython.display import display

    is_iterator = False
    if size is None:
        try:
            size = len(sequence)
        except TypeError:
            is_iterator = True
    if size is not None:
        if every is None:
            if size <= 200:
                every = 1
            else:
                every = int(size / 200)     # every 0.5%
    else:
        assert every is not None, 'sequence is iterator, set every'

    if is_iterator:
        progress = IntProgress(min=0, max=1, value=1)
        progress.bar_style = 'info'
    else:
        progress = IntProgress(min=0, max=size, value=0)
    label = HTML()
    box = VBox(children=[label, progress])
    display(box)

    index = 0
    try:
        for index, record in enumerate(sequence, 1):
            if index == 1 or index % every == 0:
                if is_iterator:
                    label.value = '{name}: {index} / ?'.format(
                        name=name,
                        index=index
                    )
                else:
                    progress.value = index
                    label.value = u'{name}: {index} / {size}'.format(
                        name=name,
                        index=index,
                        size=size
                    )
            yield record
    except:
        progress.bar_style = 'danger'
        raise
    else:
        progress.bar_style = 'success'
        progress.value = index
        label.value = "{name}: {index}".format(
            name=name,
            index=str(index or '?')
        )


def find_outliers(y, std_thr = 5):
    # 计算观测值与前后数据点的差
    diff_y_prev = np.diff(y, prepend=np.nan)
    diff_y_next = np.diff(y, append=np.nan)
    # 计算标准差的std_thr倍
    threshold = std_thr * np.std(y)
    # 找到差值都大于std_thr倍标准差的数据点的索引
    outliers_indices = np.where((np.abs(diff_y_prev) > threshold) & (np.abs(diff_y_next) > threshold))
    return outliers_indices


def denoise(y, std_thr = 5):
    outliers_indices = find_outliers(y, std_thr = std_thr)
    # 用相邻两个值的平均数替代异常值
    for i in outliers_indices:
        y[i] = (y[i-1] + y[i+1]) / 2
    return y


def plot_outliers(x, y, ax, label, std_thr = 5):
    outliers_indices = find_outliers(y, std_thr = std_thr)
    # 在折线图上标记异常点
    ax.plot(x[outliers_indices], y[outliers_indices], 'ko', label=label + ' Outliers', fillstyle='none', markerfacecolor='none')
    # 标出异常点的横坐标
    for i in outliers_indices[0]:
        ax.annotate(f'{x[i]+1:.0f}', (x[i], y[i]), textcoords="offset points", xytext=(0, 10), ha='center')

def plotFluor(path, trial):
    # trial 格式类似于 "20230904-225959"
    # path 是刺激文件夹，包含raw和process子文件夹

    import pandas as pd
    import os

    file_405 = trial + "-405-Values.csv"
    file_470 = trial + "-470-Values.csv"
    # 使用 pandas 读取 CSV 文件内容
    dat_405 = pd.read_csv(os.path.join(path, "process", file_405), header=None).values.flatten()
    dat_470 = pd.read_csv(os.path.join(path, "process", file_470), header=None).values.flatten()

    # 提取荧光值
    # 两通道可能差一张图
    n_frame = min(dat_405.shape[0], dat_470.shape[0])
    y_405 = dat_405[:n_frame]
    y_470 = dat_470[:n_frame]

    # 去掉一些串道之类的错误点
    # y_405 = denoise(y_405)
    # y_470 = denoise(y_470)

    # 计算470和405的差值
    y_470_405 = y_470 - y_405
    y_470_405_scaled = y_470_405 - np.mean(y_470_405)

    # 创建一个新的图像
    fig, axs = plt.subplots(3, 1, figsize=(12, 6), gridspec_kw={'height_ratios': [1.5, 1, 0.2]})

    # 绘制双通道荧光值
    axs[0].plot(y_405, color='purple', label='405', linewidth=0.5)
    axs[0].plot(y_470, color='#00796b', label='470', linewidth=0.5)
    plot_outliers(np.arange(n_frame), y_405, axs[0], '405')
    plot_outliers(np.arange(n_frame), y_470, axs[0], '470')
    axs[0].set_ylabel('fluor value', fontsize=12)
    axs[0].set_title('2 channel', fontsize=12)
    axs[0].legend(loc='upper right', prop={'size': 8})
    axs[0].grid(True)
    axs[0].set_xlim([0, n_frame])
    axs[0].set_xticks(range(0, n_frame, n_frame//20))

    # 绘制470-405的荧光值
    axs[1].plot(y_470_405_scaled, color='#0097a7', linewidth=0.5)
    plot_outliers(np.arange(n_frame), y_470_405_scaled, axs[1], '470-405')
    axs[1].set_ylabel('Δ fluor value', fontsize=12)
    axs[1].set_title('scaled 470-405', fontsize=12)
    axs[1].grid(True)
    axs[1].set_xlim([0, n_frame])
    axs[1].set_xticks(range(0, n_frame, n_frame//20))

    # 读取刺激时间
    stimfile = pd.read_csv(os.path.join(path, trial + ".csv"), header=None).values
    stim_delay = pd.read_csv(os.path.join(path,  trial + "-470Timestamp.csv"), header=None).values
    stim_delay = int(stim_delay[0] / 100)
    n_stimframe = min(stimfile.shape[0]//100-1, n_frame)
    stim = np.zeros(n_stimframe)
    for i in range(n_stimframe):
        stim[i] = stimfile[(i * 100 + stim_delay), 0]

    # 绘制刺激时间
    axs[2].fill_between(range(n_stimframe), stim, color='k')
    axs[2].set_xlim([0, n_frame])
    axs[2].axis('off')
    axs[2].annotate('trigger', xy=(-0.1, 0.5), xycoords='axes fraction',
                    fontsize=12, rotation=0, va='center')

    # 添加总标题
    plt.suptitle(trial, fontsize=16)
    plt.subplots_adjust(hspace=0.6)  # 调整子图之间的垂直间隔
    fig.set_facecolor('white')

    # 保存图像为PNG格式
    outputFileName = trial + "_value_stimu.png"
    plt.savefig(os.path.join(path, "process", outputFileName), dpi=300, bbox_inches='tight')
    plt.show()


# %%
