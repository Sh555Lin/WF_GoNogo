#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 19 11:00:24 2025

@author: yatangli
"""


from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import os, pickle
import pandas as pd
import sys
from PIL import Image, ImageDraw, ImageFont
from tifffile import imwrite, imread
import pathlib
from skimage.draw import polygon
import warnings
from scipy.stats import f
import json

abspath = os.path.abspath(__file__)
current_dir = os.path.dirname(abspath)
parent_dir = os.path.dirname(current_dir)
os.chdir(parent_dir)
if parent_dir not in sys.path: 
    sys.path.append(parent_dir)
from utils.wf_utils import *
# import pickle
#%%
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
                ax.set_ylabel(f"{region}\n z-score", fontsize=10)
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



#%%
base_dir = "/Volumes/Data_attention/Transfer learning/LinShu/DATA_linshu/000 Widefield"
mouse_id = "A095"
# ccf_json_path = os.path.join(base_dir,mouse_id,'20250815/process/ccf_transform.json')
dates_ls = ['20250827', '20250828', '20250829', '20250901', '20250902',
            '20250903', '20250904', '20250905', '20250923', '20250924',
            '20250925', '20250926']
date_file = os.path.join(base_dir, mouse_id,'dates.csv')
if os.path.exists(date_file):
    dates_ls = np.loadtxt(os.path.join(base_dir, mouse_id,'dates.csv'),dtype=str, delimiter=',')
else:
    np.savetxt(os.path.join(base_dir, mouse_id,'dates.csv'), dates_ls, fmt='%s',delimiter=',')


#%%
# Fix the SI calculation errors and add new difference plots
n_date = len(dates_ls)
stim_onset = 1.25 
stim_duration = 3
feedback_onset = 2.25
wf_sf = 10  # Hz
stim_onset_idx = int(stim_onset * wf_sf)
stim_offset_idx = int((stim_onset + stim_duration) * wf_sf)

# Initialize arrays for storing results
amp_left_VISp_Hit = np.zeros(n_date)
amp_right_VISp_Hit = np.zeros(n_date)
amp_left_VISp_FA = np.zeros(n_date)
amp_right_VISp_FA = np.zeros(n_date)
amp_left_VISp_CR = np.zeros(n_date)
amp_right_VISp_CR = np.zeros(n_date)
amp_left_VISp_Miss = np.zeros(n_date)
amp_right_VISp_Miss = np.zeros(n_date)

# Difference between left and right
diff_lr_VISp_Hit = np.zeros(n_date)
diff_lr_VISp_FA = np.zeros(n_date)
diff_lr_VISp_CR = np.zeros(n_date)
diff_lr_VISp_Miss = np.zeros(n_date)

# Symmetry index
si_lr_VISp_Hit = np.zeros(n_date)
si_lr_VISp_FA = np.zeros(n_date)
si_lr_VISp_CR = np.zeros(n_date)
si_lr_VISp_Miss = np.zeros(n_date)

# New: Differences between conditions within each hemisphere
diff_Hit_Miss_left = np.zeros(n_date)   # Hit - Miss for left VISp
diff_Hit_Miss_right = np.zeros(n_date)  # Hit - Miss for right VISp
diff_FA_CR_left = np.zeros(n_date)      # FA - CR for left VISp
diff_FA_CR_right = np.zeros(n_date)     # FA - CR for right VISp

type_list = ['Hit', 'Miss', 'FA', 'CR']

for i_date, date in enumerate(dates_ls):
    file_alignment = os.path.join(base_dir, mouse_id, date, 'process/ccf_transform.json')
    
    with open(file_alignment, 'r') as f:
        ccf_data = json.load(f)
    
    n_region = len(ccf_data['ccf_regions'])
    
    # Find VISp index
    visp_idx = -1
    for i in range(n_region):
        if ccf_data['ccf_regions'][i]['acronym'] == 'VISp':
            visp_idx = i
            break
    
    if visp_idx == -1:
        print(f"Warning: VISp not found in {date}")
        continue
    
    # Process each trial type
    for tp_idx, tp in enumerate(type_list):
        _file_df = os.path.join(base_dir, mouse_id, date, 
                               f"{mouse_id}_{date}_dff_mean_df_{tp}.csv")
        
        if os.path.exists(_file_df):
            df_mean = pd.read_csv(_file_df)
        else:
            _file_tif = os.path.join(base_dir, mouse_id, date, 
                                    f"{mouse_id}_{date}_dff_mean_{tp}.tif")
            
            if os.path.exists(_file_tif):
                _dff = imread(_file_tif)
                _dff_norm = norm_x(_dff)
            else:
                print(f"Warning: File not found for {date} {tp}")
                _dff_norm = np.zeros((53, 512, 512))
            
            # Create DataFrame with region data
            columns = []
            for i in range(n_region):
                region = ccf_data['ccf_regions'][i]['acronym']
                columns.append(f"{region}_l")
                columns.append(f"{region}_r")
            
            df_mean = pd.DataFrame(columns=columns)
            
            for i in range(n_region):
                _data = ccf_data['ccf_regions'][i]
                region = _data['acronym']
                
                # Left hemisphere
                _left_x = np.array(_data['left_x'])
                _left_y = np.array(_data['left_y'])
                _temporal_resp_l, _mask_l = extract_region_mean_response(_left_x, _left_y, _dff_norm)
                if len(_temporal_resp_l) > 0:
                    baseline_mean = np.mean(_temporal_resp_l[:stim_onset_idx]) if stim_onset_idx > 0 else 0
                    baseline_std = np.std(_temporal_resp_l[:stim_onset_idx]) if stim_onset_idx > 0 else 1
                    df_mean[f"{region}_l"] = (_temporal_resp_l - baseline_mean) / baseline_std
                
                # Right hemisphere
                _right_x = np.array(_data['right_x'])
                _right_y = np.array(_data['right_y'])
                _temporal_resp_r, _mask_r = extract_region_mean_response(_right_x, _right_y, _dff_norm)
                if len(_temporal_resp_r) > 0:
                    baseline_mean = np.mean(_temporal_resp_r[:stim_onset_idx]) if stim_onset_idx > 0 else 0
                    baseline_std = np.std(_temporal_resp_r[:stim_onset_idx]) if stim_onset_idx > 0 else 1
                    df_mean[f"{region}_r"] = (_temporal_resp_r - baseline_mean) / baseline_std
            
            df_mean.to_csv(_file_df, index=False)
        
        # Extract VISp amplitudes
        if 'VISp_l' in df_mean.columns and 'VISp_r' in df_mean.columns:
            # Get amplitude during stimulus period
            visp_l_stim = df_mean['VISp_l'].iloc[stim_onset_idx:stim_offset_idx+1]
            visp_r_stim = df_mean['VISp_r'].iloc[stim_onset_idx:stim_offset_idx+1]
            
            amp_l = np.mean(visp_l_stim)
            amp_r = np.mean(visp_r_stim)
            
            # Store amplitudes
            if tp == 'Hit':
                amp_left_VISp_Hit[i_date] = amp_l
                amp_right_VISp_Hit[i_date] = amp_r
                diff_lr_VISp_Hit[i_date] = (amp_l - amp_r)
                si_lr_VISp_Hit[i_date] = (amp_l - amp_r) / (abs(amp_l) + abs(amp_r) + 1e-10)
            elif tp == 'FA':
                amp_left_VISp_FA[i_date] = amp_l
                amp_right_VISp_FA[i_date] = amp_r
                diff_lr_VISp_FA[i_date] = (amp_l - amp_r) 
                si_lr_VISp_FA[i_date] = (amp_l - amp_r) / (abs(amp_l) + abs(amp_r) + 1e-10)  # Fixed: was si_lr_VISp_Hit
            elif tp == 'CR':
                amp_left_VISp_CR[i_date] = amp_l
                amp_right_VISp_CR[i_date] = amp_r
                diff_lr_VISp_CR[i_date] = (amp_l - amp_r)
                si_lr_VISp_CR[i_date] = (amp_l - amp_r) / (abs(amp_l) + abs(amp_r) + 1e-10)  # Fixed: was si_lr_VISp_Hit
            elif tp == 'Miss':
                amp_left_VISp_Miss[i_date] = amp_l
                amp_right_VISp_Miss[i_date] = amp_r
                diff_lr_VISp_Miss[i_date] = (amp_l - amp_r)
                si_lr_VISp_Miss[i_date] = (amp_l - amp_r) / (abs(amp_l) + abs(amp_r) + 1e-10)  # Fixed: was si_lr_VISp_Hit

# Calculate differences between conditions within each hemisphere
diff_Hit_Miss_left = amp_left_VISp_Hit - amp_left_VISp_Miss
diff_Hit_Miss_right = amp_right_VISp_Hit - amp_right_VISp_Miss
diff_FA_CR_left = amp_left_VISp_FA - amp_left_VISp_CR
diff_FA_CR_right = amp_right_VISp_FA - amp_right_VISp_CR

# Convert dates to MMDD format for plotting
date_objects = pd.to_datetime(dates_ls, format='%Y%m%d')
dates_mmdd = [d.strftime('%m/%d') for d in date_objects]

# =====================================================================
# FIGURE 1: Individual Trial Type Plots (2x2 grid)
# =====================================================================
print("\nFIGURE 1: VISp Amplitudes by Trial Type")
fig1, axes1 = plt.subplots(2, 2, figsize=(15, 8))
fig1.suptitle(f'{mouse_id} - VISp Activity by Trial Type', fontsize=16, fontweight='bold')

trial_configs = [
    ('Hit', amp_left_VISp_Hit, amp_right_VISp_Hit, 0, 0),
    ('FA', amp_left_VISp_FA, amp_right_VISp_FA, 0, 1),
    ('CR', amp_left_VISp_CR, amp_right_VISp_CR, 1, 0),
    ('Miss', amp_left_VISp_Miss, amp_right_VISp_Miss, 1, 1)
]

for trial_name, left_data, right_data, row, col in trial_configs:
    ax = axes1[row, col]
    ax.plot(dates_mmdd, left_data, 'bo-', label='Left', linewidth=2, markersize=6)
    ax.plot(dates_mmdd, right_data, 'ro-', label='Right', linewidth=2, markersize=6)
    ax.set_title(f'{trial_name} Trials', fontsize=14)
    ax.set_ylabel('Amplitude (z)', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)
    ax.set_xticks(range(len(dates_mmdd)))
    ax.set_xticklabels(dates_mmdd, rotation=45, ha='right')

plt.tight_layout()
plt.show()

# =====================================================================
# FIGURE 2: Combined Bar Plots
# =====================================================================
print("\nFIGURE 2: VISp Combined Comparison Plots")
fig2, axes2 = plt.subplots(2, 2, figsize=(15, 8))
fig2.suptitle(f'{mouse_id} - VISp Amplitudes and Differences', fontsize=16, fontweight='bold')

x_pos = np.arange(len(dates_mmdd))

# Top row: All trial types bar plots
bar_width = 0.18
colors = {'Hit': 'green', 'FA': 'red', 'CR': 'blue', 'Miss': 'orange'}

# Top-left: Left VISp all trial types
ax = axes2[0, 0]
for i, (trial_name, color) in enumerate(colors.items()):
    offset = (i - 1.5) * bar_width
    data = [amp_left_VISp_Hit, amp_left_VISp_FA, amp_left_VISp_CR, amp_left_VISp_Miss][i]
    ax.bar(x_pos + offset, data, bar_width, label=trial_name, color=color, alpha=0.7)
ax.set_title('A. Left VISp: All Trial Types', fontsize=14)
ax.set_ylabel('Amplitude (z)', fontsize=11)
ax.legend(fontsize=9, ncol=2)
ax.grid(True, alpha=0.2, axis='y')
ax.set_xticks(x_pos)
ax.set_xticklabels(dates_mmdd, rotation=45, ha='right')

# Top-right: Right VISp all trial types
ax = axes2[0, 1]
for i, (trial_name, color) in enumerate(colors.items()):
    offset = (i - 1.5) * bar_width
    data = [amp_right_VISp_Hit, amp_right_VISp_FA, amp_right_VISp_CR, amp_right_VISp_Miss][i]
    ax.bar(x_pos + offset, data, bar_width, label=trial_name, color=color, alpha=0.7)
ax.set_title('B. Right VISp: All Trial Types', fontsize=14)
ax.set_ylabel('Amplitude (z)', fontsize=11)
ax.legend(fontsize=9, ncol=2)
ax.grid(True, alpha=0.2, axis='y')
ax.set_xticks(x_pos)
ax.set_xticklabels(dates_mmdd, rotation=45, ha='right')

# Bottom row: Condition differences bar plots
bar_width_combined = 0.35  # Wider for two bars

# Bottom-left: Combined Hit-Miss & FA-CR for Left VISp
ax = axes2[1, 0]
bars1 = ax.bar(x_pos - bar_width_combined/2, diff_Hit_Miss_left, bar_width_combined, 
               label='Hit - Miss', color='purple', alpha=0.7, edgecolor='black')
bars2 = ax.bar(x_pos + bar_width_combined/2, diff_FA_CR_left, bar_width_combined, 
               label='FA - CR', color='teal', alpha=0.7, edgecolor='black')
ax.set_title('C. Left VISp: Condition Differences', fontsize=14)
ax.set_xlabel('Date', fontsize=11)
ax.set_ylabel('Amplitude Difference', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2, axis='y')
ax.axhline(y=0, color='black', alpha=0.3, linestyle='--')
ax.set_xticks(x_pos)
ax.set_xticklabels(dates_mmdd, rotation=45, ha='right')

# Bottom-right: Combined Hit-Miss & FA-CR for Right VISp
ax = axes2[1, 1]
bars1 = ax.bar(x_pos - bar_width_combined/2, diff_Hit_Miss_right, bar_width_combined, 
               label='Hit - Miss', color='magenta', alpha=0.7, edgecolor='black')
bars2 = ax.bar(x_pos + bar_width_combined/2, diff_FA_CR_right, bar_width_combined, 
               label='FA - CR', color='cyan', alpha=0.7, edgecolor='black')
ax.set_title('D. Right VISp: Condition Differences', fontsize=14)
ax.set_xlabel('Date', fontsize=11)
ax.set_ylabel('Amplitude Difference', fontsize=11)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.2, axis='y')
ax.axhline(y=0, color='black', alpha=0.3, linestyle='--')
ax.set_xticks(x_pos)
ax.set_xticklabels(dates_mmdd, rotation=45, ha='right')

plt.tight_layout()
plt.show()

# =====================================================================
# FIGURE 3: Left-Right Differences (Line Plots)
# =====================================================================
print("\nFIGURE 3: VISp Left-Right Differences")
fig3, axes3 = plt.subplots(1, 2, figsize=(15, 5))
fig3.suptitle(f'{mouse_id} - VISp Hemispheric Differences', fontsize=16, fontweight='bold')

# Left: Simple Left-Right Difference
ax = axes3[0]
diff_configs = [
    ('Hit', diff_lr_VISp_Hit, 'green'),
    ('FA', diff_lr_VISp_FA, 'red'),
    ('CR', diff_lr_VISp_CR, 'blue'),
    ('Miss', diff_lr_VISp_Miss, 'orange')
]

for trial_name, data, color in diff_configs:
    ax.plot(dates_mmdd, data, 'o-', label=trial_name, color=color, linewidth=2, markersize=6)

ax.set_title('A. Left-Right Difference (L - R)', fontsize=14)
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Amplitude Difference', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.2)
ax.axhline(y=0, color='black', alpha=0.3)
ax.set_xticks(range(len(dates_mmdd)))
ax.set_xticklabels(dates_mmdd, rotation=45, ha='right')

# Right: Symmetry Index
ax = axes3[1]
si_configs = [
    ('Hit', si_lr_VISp_Hit, 'green'),
    ('FA', si_lr_VISp_FA, 'red'),
    ('CR', si_lr_VISp_CR, 'blue'),
    ('Miss', si_lr_VISp_Miss, 'orange')
]

for trial_name, data, color in si_configs:
    ax.plot(dates_mmdd, data, 'o-', label=trial_name, color=color, linewidth=2, markersize=6)

ax.set_title('B. Symmetry Index (L-R)/(|L|+|R|)', fontsize=14)
ax.set_xlabel('Date', fontsize=12)
ax.set_ylabel('Symmetry Index', fontsize=12)
ax.legend(fontsize=10)
ax.grid(True, alpha=0.2)
ax.axhline(y=0, color='black', alpha=0.3)
ax.set_xticks(range(len(dates_mmdd)))
ax.set_xticklabels(dates_mmdd, rotation=45, ha='right')

plt.tight_layout()
plt.show()

#%%


n_date = dates_ls.size
stim_onset = 1.25 
stim_duration = 3
feedback_onset = 2.25
diff_lr_VISp_Hit = np.zeros(n_date)
diff_lr_VISp_FA = np.zeros(n_date)
diff_lr_VISp_CR = np.zeros(n_date)
diff_lr_VISp_Miss = np.zeros(n_date)
amp_left_VISp_Hit = np.zeros(n_date)
diff_lr_VISp_FA = np.zeros(n_date)
diff_lr_VISp_CR = np.zeros(n_date)
diff_lr_VISp_Miss = np.zeros(n_date)
diff_lr_VISp_Hit = np.zeros(n_date)
diff_lr_VISp_FA = np.zeros(n_date)
diff_lr_VISp_CR = np.zeros(n_date)
diff_lr_VISp_Miss = np.zeros(n_date)
for i_date, date in enumerate(dates_ls):
    #%
    # date = date_ls[1]
    # with open(ccf_json_path, 'r') as f:
    #     ccf_data = json.load(f)
    file_alignment = os.path.join(base_dir, mouse_id, date,'process/ccf_transform.json')#'widefield_alignment/wf_alignment_A095.pkl')
    # Read a pickle file
    # with open(file_alignment, 'rb') as f:  # 'rb' means read binary
    #     data = load_pickle_safely(f)
    with open(file_alignment, 'r') as f:
        ccf_data = json.load(f)
        
    
    # plot V1 to check the alignment 
    n_region = len(ccf_data['ccf_regions'])   
    # for i in range(n_region):
    #     if ccf_data['ccf_regions'][i]['acronym']=='VISp':
    #         print(i)
    #         _data = ccf_data['ccf_regions'][i]
    #         _left_x = np.array(_data['left_x'])
    #         _left_y = np.array(_data['left_y'])
    #         plt.plot(_left_x,_left_y,'.')
    #         _right_x = np.array(_data['right_x'])
    #         _right_y = np.array(_data['right_y'])
    #         plt.plot(_right_x,_right_y,'.')
    #         plt.axis('equal') 
    #         plt.show()
        
    
    
#% 

    #% generate eight dataframes: (Average, single trial) X (Hit, Miss, FA, CR)
    
    type_list = ['Hit','Miss', 'FA','CR']
    wf_sf = 10 #Hz
    stim_onset = 1.25 
    idx_onset = int(stim_onset*wf_sf)
    stim_duration = 3
    feedback_onset = 2.25
    
    stim_onset_idx = int(stim_onset*wf_sf)
    stim_offset_idx = int((stim_onset+stim_duration)*wf_sf)
    
    columns = []
    for i in range(n_region):
        region = ccf_data['ccf_regions'][i]['acronym']
        columns.append(f"{region}_l")
        columns.append(f"{region}_r")
    
    df_mean_ls = []   
    for tp in type_list:     
        _file_df = os.path.join(base_dir, mouse_id,date,mouse_id+'_'+date+'_dff_mean_df_'+tp+'.csv')
        if os.path.exists(_file_df):
            df_mean = pd.read_csv(_file_df) 
        else:
            _file_tif = os.path.join(base_dir, mouse_id,date, mouse_id+'_'+date+'_dff_mean_'+tp+'.tif')
            if os.path.exists(_file_tif):
                _dff = imread(_file_tif)
                _dff_norm = norm_x(_dff)
                _mean = np.mean(_dff_norm[:idx_onset,:,:],axis=0)
                _sd = np.std(_dff_norm[:idx_onset,:,:],axis=0)
                _dff_zs = np.zeros_like(_dff)
                for i in range(_dff.shape[0]):
                    _dff_zs[i,:,:] = (_dff_norm[i,:,:] -  _mean)/_sd 
            else:
                _dff_norm = np.zeros((53,512,512))
            df_mean = pd.DataFrame(columns=columns)
            for i in range(n_region):
                _data = ccf_data['ccf_regions'][i]
                region = _data['acronym']
                _left_x = np.array(_data['left_x'])
                _left_y = np.array(_data['left_y'])
                _temporal_resp, _mask = extract_region_mean_response(_left_x,_left_y,_dff_norm)
                _l = region+'_l'
                df_mean[_l] = (_temporal_resp-_temporal_resp[:idx_onset].mean())/_temporal_resp[:idx_onset].std()
                
                _right_x = np.array(_data['right_x'])
                _right_y = np.array(_data['right_y'])
                _temporal_resp, _mask = extract_region_mean_response(_right_x,_right_y,_dff_norm)
                _r = region+'_r'
                df_mean[_r] = (_temporal_resp-_temporal_resp[:idx_onset].mean())/_temporal_resp[:idx_onset].std()
            df_mean.to_csv(_file_df,index=False)     
        df_mean_ls.append(df_mean)
        
        _l = np.mean(df_mean['VISp_l'][stim_onset_idx:stim_offset_idx+1])
        _r = np.mean(df_mean['VISp_r'][stim_onset_idx:stim_offset_idx+1])
        if tp == 'Hit':
            diff_lr_VISp_Hit[i_date] = (_l-_r)/(_l+_r)
        if tp == 'FA':
            diff_lr_VISp_FA[i_date] = (_l-_r)/(_l+_r)
    
    fig_file = os.path.join(base_dir, mouse_id,date,'fig_temporal_profiles')
    if not os.path.exists(fig_file+'.png'):
        plot_brain_region_temporal_profiles(df_mean_ls,type_list,fig_file=fig_file,title=mouse_id+':'+date)


date_objects = pd.to_datetime(dates_ls, format='%Y%m%d')
dates_mmdd = [d.strftime('%m%d') for d in date_objects]

plt.plot(dates_mmdd ,diff_lr_VISp_Hit,'k*-')
plt.title('Hit')
plt.show()

plt.plot(dates_mmdd,diff_lr_VISp_FA,'ro-')
plt.title('FA')
plt.show()
#%%
conditions = type_list
results_latency = analyze_all_conditions(df_mean_ls, conditions, fs=10, measure='latency')
results_amplitude = analyze_all_conditions(df_mean_ls, conditions, fs=10, measure='amp')

#%%
results, p_vals, lags = analyze_all_causality(
    df_mean_ls, conditions, causality_type='granger', lag='auto')


results, p_vals, lags = analyze_all_causality(
    df_mean_ls, conditions, causality_type='transfer', lag=1)

#%%
# #%%
# import numpy as np
# import pandas as pd
# from statsmodels.tsa.stattools import grangercausalitytests
# import matplotlib.pyplot as plt
# from scipy import signal
# from scipy.stats import circmean, circstd
# import networkx as nx

# def granger_causality_analysis(time_series_dict, maxlag=10):
#     """
#     Perform Granger causality between all pairs of regions.
    
#     Parameters:
#     -----------
#     time_series_dict : dict
#         Dictionary of time series: {'region1': ts1, 'region2': ts2, ...}
#     maxlag : int
#         Maximum number of lags to test
        
#     Returns:
#     --------
#     causality_matrix : pandas DataFrame
#         Matrix of p-values for Granger causality tests
#     """
#     regions = list(time_series_dict.keys())
#     n_regions = len(regions)
    
#     # Create matrix to store p-values
#     p_values = np.ones((n_regions, n_regions))
    
#     # Test all pairs
#     for i, region_i in enumerate(regions):
#         for j, region_j in enumerate(regions):
#             if i != j:
#                 # Combine time series: region_j is Y (dependent), region_i is X (independent)
#                 # Testing if region_i Granger-causes region_j
#                 data = np.column_stack([time_series_dict[region_j], 
#                                        time_series_dict[region_i]])
                
#                 # Granger test (region_i -> region_j)
#                 try:
#                     result = grangercausalitytests(data, maxlag=maxlag, verbose=False)
                    
#                     # Get minimum p-value across lags
#                     p_vals = []
#                     for lag in range(1, maxlag + 1):
#                         # Different test statistics available
#                         # Using ssr_ftest (F-test based on residual sum of squares)
#                         p_val = result[lag][0]['ssr_ftest'][1]
#                         p_vals.append(p_val)
                    
#                     p_values[i, j] = min(p_vals) if p_vals else 1.0
                    
#                 except Exception as e:
#                     # Handle cases where test fails (e.g., insufficient data)
#                     print(f"Granger test failed for {region_i}->{region_j}: {str(e)[:50]}")
#                     p_values[i, j] = 1.0
    
#     # Create DataFrame
#     causality_df = pd.DataFrame(p_values, index=regions, columns=regions)
    
#     return causality_df


# def plot_causality_network(causality_df, threshold=0.05):
#     """
#     Plot causal network.
#     """

    
#     G = nx.DiGraph()
    
#     # Add nodes
#     for region in causality_df.columns:
#         G.add_node(region)
    
#     # Add edges based on significance
#     for i, source in enumerate(causality_df.columns):
#         for j, target in enumerate(causality_df.columns):
#             if i != j and causality_df.iloc[i, j] < threshold:
#                 # Add edge with weight = -log10(p-value)
#                 weight = -np.log10(causality_df.iloc[i, j])
#                 G.add_edge(source, target, weight=weight)
    
#     # Plot network
#     plt.figure(figsize=(10, 8))
    
#     # Use spring layout with adjusted parameters
#     pos = nx.spring_layout(G, k=2, iterations=100, seed=42)
    
#     # Draw nodes
#     nx.draw_networkx_nodes(G, pos, node_size=800, 
#                           node_color='lightblue', alpha=0.9, edgecolors='black')
    
#     # Draw edges with thickness based on weight
#     edges = G.edges()
#     if edges:
#         weights = [G[u][v]['weight'] for u, v in edges]
#         min_w, max_w = min(weights), max(weights)
        
#         # Normalize weights for line width
#         if max_w > min_w:
#             widths = [2 + 8 * (w - min_w) / (max_w - min_w) for w in weights]
#         else:
#             widths = [3] * len(weights)
        
#         nx.draw_networkx_edges(G, pos, edgelist=edges,
#                               width=widths,
#                               edge_color='red', alpha=0.7,
#                               arrows=True, arrowsize=20, 
#                               arrowstyle='->', connectionstyle='arc3,rad=0.1')
    
#     # Draw labels
#     nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
    
#     plt.title(f'Granger Causality Network (p < {threshold})', fontsize=14, fontweight='bold')
#     plt.axis('off')
#     plt.tight_layout()
#     plt.show()
    
#     return G


# try:
#     from pyitlib import discrete_random_variable as drv
#     PYITLIB_AVAILABLE = True
# except ImportError:
#     print("Warning: pyitlib not installed. Install with: pip install pyitlib")
#     PYITLIB_AVAILABLE = False


# def transfer_entropy_analysis(time_series_dict, bin_method='uniform', n_bins=10):
#     """
#     Calculate transfer entropy between brain regions.
    
#     Transfer entropy measures: TE(X->Y) = I(Y_future | Y_past, X_past) - I(Y_future | Y_past)
#     """
#     if not PYITLIB_AVAILABLE:
#         print("Error: pyitlib is required for transfer entropy analysis")
#         return pd.DataFrame(), {}
    
#     regions = list(time_series_dict.keys())
#     n_regions = len(regions)
    
#     # Discretize time series
#     discretized = {}
#     for region, ts in time_series_dict.items():
#         ts_array = np.array(ts)
#         if bin_method == 'uniform':
#             # Uniform binning
#             bins = np.linspace(np.min(ts_array), np.max(ts_array), n_bins + 1)
#             discretized[region] = np.digitize(ts_array, bins) - 1  # 0-indexed
#         elif bin_method == 'quantile':
#             # Quantile binning
#             percentiles = np.linspace(0, 100, n_bins + 1)
#             bin_edges = np.percentile(ts_array, percentiles)
#             discretized[region] = np.digitize(ts_array, bin_edges) - 1
#         else:
#             raise ValueError("bin_method must be 'uniform' or 'quantile'")
    
#     # Calculate transfer entropy matrix
#     te_matrix = np.zeros((n_regions, n_regions))
    
#     for i, region_i in enumerate(regions):
#         for j, region_j in enumerate(regions):
#             if i != j:
#                 X = discretized[region_i].astype(int)
#                 Y = discretized[region_j].astype(int)
                
#                 # Ensure minimum length
#                 if len(X) < 3 or len(Y) < 3:
#                     te_matrix[i, j] = 0
#                     continue
                
#                 # Create past and future vectors (1-step lag)
#                 X_past = X[:-1]
#                 Y_past = Y[:-1]
#                 Y_future = Y[1:]
                
#                 # Calculate transfer entropy: TE(X->Y)
#                 try:
#                     # TE = I(Y_future; X_past | Y_past)
#                     te = drv.information_mutual_conditional(Y_future, X_past, Y_past)
#                     te_matrix[i, j] = te
#                 except Exception as e:
#                     # Handle calculation errors
#                     print(f"TE calculation failed for {region_i}->{region_j}: {str(e)[:50]}")
#                     te_matrix[i, j] = 0
    
#     # Create DataFrame
#     te_df = pd.DataFrame(te_matrix, index=regions, columns=regions)
    
#     return te_df, discretized


# def analyze_information_flow(te_matrix, direction='outgoing'):
#     """
#     Analyze information flow patterns.
#     """
#     if direction == 'outgoing':
#         # Which regions send most information (sum over columns they point to)
#         flow_measure = te_matrix.sum(axis=1)  # Sum across columns (outgoing)
#     elif direction == 'incoming':
#         # Which regions receive most information (sum over rows pointing to them)
#         flow_measure = te_matrix.sum(axis=0)  # Sum across rows (incoming)
#     elif direction == 'net':
#         # Net flow (outgoing - incoming)
#         flow_measure = te_matrix.sum(axis=1) - te_matrix.sum(axis=0)
#     else:
#         raise ValueError("direction must be 'outgoing', 'incoming', or 'net'")
    
#     return pd.Series(flow_measure, index=te_matrix.index).sort_values(ascending=False)


# def cross_correlation_lag_analysis(time_series_dict, max_lag=10):
#     """
#     Find time lags between regions using cross-correlation.
#     """
#     regions = list(time_series_dict.keys())
#     n_regions = len(regions)
    
#     lag_matrix = np.zeros((n_regions, n_regions))
#     corr_matrix = np.zeros((n_regions, n_regions))
    
#     for i, region_i in enumerate(regions):
#         ts_i = np.array(time_series_dict[region_i])
#         mean_i = np.mean(ts_i)
#         ts_i_centered = ts_i - mean_i
        
#         for j, region_j in enumerate(regions):
#             if i != j:
#                 ts_j = np.array(time_series_dict[region_j])
#                 mean_j = np.mean(ts_j)
#                 ts_j_centered = ts_j - mean_j
                
#                 # Cross-correlation with proper normalization
#                 correlation = np.correlate(ts_i_centered, ts_j_centered, mode='full')
                
#                 # Normalize by standard deviations
#                 norm = np.sqrt(np.sum(ts_i_centered**2) * np.sum(ts_j_centered**2))
#                 if norm > 0:
#                     correlation = correlation / norm
                
#                 # Limit to reasonable lags
#                 n = len(ts_i)
                
#                 valid_lags = np.arange(-min(max_lag, n-1), min(max_lag, n-1)+1)
#                 valid_corr = correlation[n-1-min(max_lag, n-1):n+max_lag]
                
#                 # Find peak correlation
#                 peak_idx = np.argmax(np.abs(valid_corr))
#                 lag_matrix[i, j] = valid_lags[peak_idx]
#                 corr_matrix[i, j] = valid_corr[peak_idx]
#     lag_matrix[lag_matrix==max_lag] = np.nan
#     lag_matrix[lag_matrix==-max_lag] = np.nan
#     # Create DataFrames
#     lag_df = pd.DataFrame(lag_matrix, index=regions, columns=regions)
#     corr_df = pd.DataFrame(corr_matrix, index=regions, columns=regions)
    
#     # Determine direction based on lags
#     direction_df = pd.DataFrame('', index=regions, columns=regions)
#     for i in range(n_regions):
#         for j in range(n_regions):
#             if i != j:
#                 if lag_df.iloc[i, j] > 0:
#                     direction_df.iloc[i, j] = f"{regions[i]} → {regions[j]}"
#                 elif lag_df.iloc[i, j] < 0:
#                     direction_df.iloc[i, j] = f"{regions[j]} → {regions[i]}"
#                 else:
#                     direction_df.iloc[i, j] = "No lag"
    
#     return lag_df, corr_df, direction_df


# # def phase_based_connectivity(time_series_dict, fs=1000, freq_band=(8, 12)):
# #     """
# #     Analyze phase-based connectivity (e.g., Phase Locking Value).
# #     """
# #     regions = list(time_series_dict.keys())
# #     n_regions = len(regions)
    
# #     # Hilbert transform to get instantaneous phase
# #     phases = {}
# #     for region, ts in time_series_dict.items():
# #         ts_array = np.array(ts)
        
# #         # Bandpass filter
# #         nyquist = fs / 2
# #         low = freq_band[0] / nyquist
# #         high = freq_band[1] / nyquist
        
# #         if low < 1 and high < 1:
# #             b, a = signal.butter(4, [low, high], 'bandpass')
# #             filtered = signal.filtfilt(b, a, ts_array)
            
# #             # Hilbert transform
# #             analytic_signal = signal.hilbert(filtered)
# #             phases[region] = np.angle(analytic_signal)
# #         else:
# #             # If frequency band is invalid, use raw phase
# #             analytic_signal = signal.hilbert(ts_array)
# #             phases[region] = np.angle(analytic_signal)
    
# #     # Calculate Phase Locking Value (PLV)
# #     plv_matrix = np.zeros((n_regions, n_regions))
    
# #     for i, region_i in enumerate(regions):
# #         for j, region_j in enumerate(regions):
# #             if i != j:
# #                 phase_diff = phases[region_i] - phases[region_j]
# #                 plv = np.abs(np.mean(np.exp(1j * phase_diff)))
# #                 plv_matrix[i, j] = plv
    
# #     # Calculate phase slope index (PSI) for directionality
# #     psi_matrix = np.zeros((n_regions, n_regions))
    
# #     for i, region_i in enumerate(regions):
# #         for j, region_j in enumerate(regions):
# #             if i != j:
# #                 # Cross-spectrum
# #                 f, Cxy = signal.csd(time_series_dict[region_i], 
# #                                    time_series_dict[region_j], 
# #                                    fs=fs, nperseg=min(256, len(time_series_dict[region_i])//4))
                
# #                 # Phase slope
# #                 freq_idx = np.where((f >= freq_band[0]) & (f <= freq_band[1]))[0]
# #                 if len(freq_idx) > 1:
# #                     phase = np.angle(Cxy[freq_idx])
# #                     # PSI calculation
# #                     psi = np.sum(np.conj(phase[:-1]) * phase[1:])
# #                     psi_matrix[i, j] = np.imag(psi)  # Imaginary part indicates direction
# #                 else:
# #                     psi_matrix[i, j] = 0
    
# #     return pd.DataFrame(plv_matrix, index=regions, columns=regions), \
# #            pd.DataFrame(psi_matrix, index=regions, columns=regions)


# def find_consensus_connections(results, threshold=0.7):
#     """
#     Find connections that appear consistently across methods.
#     """
#     regions = list(results['granger_causality'].index)
#     n_regions = len(regions)
    
#     # Normalize each matrix
#     normalized_matrices = []
    
#     # Granger (1-p value, so higher = more significant)
#     gc_norm = 1 - results['granger_causality'].values
#     gc_min, gc_max = gc_norm.min(), gc_norm.max()
#     if gc_max > gc_min:
#         gc_norm = (gc_norm - gc_min) / (gc_max - gc_min)
#     normalized_matrices.append(gc_norm)
    
#     # Transfer entropy (if available)
#     if 'transfer_entropy' in results and not results['transfer_entropy'].empty:
#         te_norm = results['transfer_entropy'].values
#         te_min, te_max = te_norm.min(), te_norm.max()
#         if te_max > te_min:
#             te_norm = (te_norm - te_min) / (te_max - te_min)
#         normalized_matrices.append(te_norm)
    
#     # Phase locking (if available)
#     if 'phase_locking' in results and not results['phase_locking'].empty:
#         plv_norm = results['phase_locking'].values
#         plv_min, plv_max = plv_norm.min(), plv_norm.max()
#         if plv_max > plv_min:
#             plv_norm = (plv_norm - plv_min) / (plv_max - plv_min)
#         normalized_matrices.append(plv_norm)
    
#     # Cross-correlation (if available)
#     if 'cross_corr' in results and not results['cross_corr'].empty:
#         cc_norm = np.abs(results['cross_corr'].values)  # Use absolute correlation
#         cc_min, cc_max = cc_norm.min(), cc_norm.max()
#         if cc_max > cc_min:
#             cc_norm = (cc_norm - cc_min) / (cc_max - cc_min)
#         normalized_matrices.append(cc_norm)
    
#     # Average across available methods
#     if normalized_matrices:
#         consensus_matrix = np.mean(normalized_matrices, axis=0)
        
#         # Threshold
#         binary_consensus = consensus_matrix > threshold
#         np.fill_diagonal(binary_consensus, False)
        
#         consensus_df = pd.DataFrame(binary_consensus.astype(int), 
#                                    index=regions, columns=regions)
#     else:
#         consensus_df = pd.DataFrame()
    
#     return consensus_df


# def analyze_information_flow_pipeline(time_series_dict, fs=1000):
#     """
#     Complete pipeline for information flow analysis.
#     """
#     results = {}
    
#     # 1. Granger Causality
#     print("1. Running Granger Causality...")
#     gc_matrix = granger_causality_analysis(time_series_dict, maxlag=20)
#     results['granger_causality'] = gc_matrix
    
#     # 2. Transfer Entropy (if pyitlib is available)
#     print("2. Calculating Transfer Entropy...")
#     if PYITLIB_AVAILABLE:
#         te_matrix, discretized = transfer_entropy_analysis(time_series_dict)
#         results['transfer_entropy'] = te_matrix
#         results['discretized_series'] = discretized
#     else:
#         print("   Skipping (pyitlib not installed)")
#         results['transfer_entropy'] = pd.DataFrame()
    
#     # # 3. Phase-based connectivity
#     # print("3. Analyzing phase-based connectivity...")
#     # plv_matrix, psi_matrix = phase_based_connectivity(time_series_dict, fs=fs)
#     # results['phase_locking'] = plv_matrix
#     # results['phase_slope'] = psi_matrix
    
#     # 4. Cross-correlation lags
#     print("4. Calculating cross-correlation lags...")
#     lag_matrix, corr_matrix, direction_matrix = cross_correlation_lag_analysis(time_series_dict)
#     results['cross_corr_lags'] = lag_matrix
#     results['cross_corr'] = corr_matrix
#     results['direction'] = direction_matrix
    
#     # 5. Consensus analysis
#     print("5. Finding consensus connections...")
#     consensus = find_consensus_connections(results)
#     results['consensus'] = consensus
    
#     return results


# def extract_region_timeseries(df_mean_ls, conditions, region, hemisphere='both'):
#     """
#     Extract time series for a specific region across conditions.
    
#     Parameters:
#     -----------
#     df_mean_ls : list of DataFrames
#         List of 4 DataFrames for ['Hit', 'Miss', 'FA', 'CR']
#     conditions : list of str
#         Condition names ['Hit', 'Miss', 'FA', 'CR']
#     region : str
#         Brain region name (e.g., 'MOp', 'VISp')
#     hemisphere : str
#         'left', 'right', or 'both' (averaged)
        
#     Returns:
#     --------
#     time_series_dict : dict
#         Dictionary of time series for each condition
#     """
#     time_series_dict = {}
    
#     for df, cond in zip(df_mean_ls, conditions):
#         if hemisphere == 'both':
#             # Average left and right hemispheres
#             left_col = f"{region}_l"
#             right_col = f"{region}_r"
            
#             if left_col in df.columns and right_col in df.columns:
#                 ts = (df[left_col] + df[right_col]) / 2
#             elif left_col in df.columns:
#                 ts = df[left_col]
#             elif right_col in df.columns:
#                 ts = df[right_col]
#             else:
#                 raise ValueError(f"Region {region} not found in DataFrame")
                
#         elif hemisphere in ['left', 'right']:
#             col_name = f"{region}_{hemisphere[0]}"  # 'l' or 'r'
#             if col_name in df.columns:
#                 ts = df[col_name]
#             else:
#                 raise ValueError(f"Region {region}_{hemisphere[0]} not found")
#         else:
#             raise ValueError("hemisphere must be 'left', 'right', or 'both'")
        
#         time_series_dict[cond] = ts.values
    
#     return time_series_dict


# def extract_all_regions_timeseries(df_mean_ls, conditions, hemisphere='both'):
#     """
#     Extract time series for ALL regions.
    
#     Returns:
#     --------
#     nested_dict : dict of dicts
#         {region_name: {condition: time_series}}
#     """
#     # Get all region names
#     all_columns = df_mean_ls[0].columns
#     region_names = sorted(list(set(col[:-2] for col in all_columns 
#                                  if col.endswith(('_l', '_r')))))
    
#     nested_dict = {}
    
#     for region in region_names:
#         try:
#             nested_dict[region] = extract_region_timeseries(
#                 df_mean_ls, conditions, region, hemisphere
#             )
#         except ValueError as e:
#             print(f"Warning: {str(e)}. Skipping region {region}.")
#             continue
    
#     return nested_dict, region_names


# def visualize_all_results(results):
#     """
#     Visualize all information flow analysis results.
#     """
#     fig = plt.figure(figsize=(18, 12))
    
#     # 1. Granger Causality heatmap
#     ax1 = plt.subplot(2, 3, 1)
#     gc_matrix = results['granger_causality']
#     # Use -log10(p-value) for better visualization
#     gc_vis = -np.log10(gc_matrix.values + 1e-10)  # Add small epsilon to avoid log(0)
#     im1 = ax1.imshow(gc_vis, cmap='hot_r')
#     ax1.set_title('Granger Causality\n(-log10 p-value)', fontsize=11, fontweight='bold')
#     ax1.set_xticks(range(len(gc_matrix.columns)))
#     ax1.set_yticks(range(len(gc_matrix.index)))
#     ax1.set_xticklabels(gc_matrix.columns, rotation=45, ha='right', fontsize=9)
#     ax1.set_yticklabels(gc_matrix.index, fontsize=9)
#     plt.colorbar(im1, ax=ax1, label='-log10(p)')
    
#     # Add significance threshold line
#     threshold = -np.log10(0.05)
#     for i in range(gc_vis.shape[0]):
#         for j in range(gc_vis.shape[1]):
#             if i != j and gc_vis[i, j] > threshold:
#                 ax1.text(j, i, '★', ha='center', va='center', 
#                         color='white', fontsize=8, fontweight='bold')
    
#     # 2. Transfer Entropy heatmap (if available)
#     ax2 = plt.subplot(2, 3, 2)
#     if not results['transfer_entropy'].empty:
#         te_matrix = results['transfer_entropy']
#         im2 = ax2.imshow(te_matrix.values, cmap='viridis')
#         ax2.set_title('Transfer Entropy', fontsize=11, fontweight='bold')
#         ax2.set_xticks(range(len(te_matrix.columns)))
#         ax2.set_yticks(range(len(te_matrix.index)))
#         ax2.set_xticklabels(te_matrix.columns, rotation=45, ha='right', fontsize=9)
#         ax2.set_yticklabels(te_matrix.index, fontsize=9)
#         plt.colorbar(im2, ax=ax2, label='TE (bits)')
        
#         # Highlight top 10% of connections
#         te_vals = te_matrix.values.copy()
#         np.fill_diagonal(te_vals, -np.inf)
#         threshold = np.percentile(te_vals[te_vals > -np.inf], 90)
#         for i in range(te_vals.shape[0]):
#             for j in range(te_vals.shape[1]):
#                 if i != j and te_vals[i, j] > threshold:
#                     ax2.text(j, i, '★', ha='center', va='center', 
#                             color='white', fontsize=8, fontweight='bold')
#     else:
#         ax2.text(0.5, 0.5, 'Transfer Entropy\n(pyitlib not installed)', 
#                 ha='center', va='center', transform=ax2.transAxes, fontsize=10)
#         ax2.axis('off')
    
#     # 3. Direction Matrix Visualization
#     ax3 = plt.subplot(2, 3, 3)
#     if 'direction' in results and not results['direction'].empty:
#         direction_matrix = results['direction']
        
#         # Create numerical matrix for visualization
#         # 1 = X→Y, -1 = Y→X, 0 = no direction/insignificant
#         dir_vals = np.zeros(direction_matrix.shape)
        
#         regions = list(direction_matrix.index)
#         for i in range(len(regions)):
#             for j in range(len(regions)):
#                 if i != j:
#                     cell_val = direction_matrix.iloc[i, j]
#                     if '→' in str(cell_val):
#                         # Check direction
#                         dir_parts = str(cell_val).split(' → ')
#                         if len(dir_parts) == 2:
#                             if dir_parts[0] == regions[i]:
#                                 dir_vals[i, j] = 1  # i→j
#                             else:
#                                 dir_vals[i, j] = -1  # j→i
        
#         # Create custom colormap for direction
#         from matplotlib.colors import ListedColormap
#         dir_cmap = ListedColormap(['red', 'gray', 'blue'])
        
#         im3 = ax3.imshow(dir_vals, cmap=dir_cmap, vmin=-1, vmax=1)
#         ax3.set_title('Information Flow Direction', fontsize=11, fontweight='bold')
#         ax3.set_xticks(range(len(direction_matrix.columns)))
#         ax3.set_yticks(range(len(direction_matrix.index)))
#         ax3.set_xticklabels(direction_matrix.columns, rotation=45, ha='right', fontsize=9)
#         ax3.set_yticklabels(direction_matrix.index, fontsize=9)
        
#         # Add custom colorbar
#         cbar = plt.colorbar(im3, ax=ax3, ticks=[-1, 0, 1])
#         cbar.ax.set_yticklabels(['j→i', 'No flow', 'i→j'])
        
#         # Add arrow annotations for significant flows
#         for i in range(len(regions)):
#             for j in range(len(regions)):
#                 if i != j and dir_vals[i, j] != 0:
#                     if dir_vals[i, j] == 1:
#                         arrow_char = '→'
#                         color = 'white'
#                     else:
#                         arrow_char = '←'
#                         color = 'black'
#                     ax3.text(j, i, arrow_char, ha='center', va='center', 
#                             color=color, fontsize=12, fontweight='bold')
#     else:
#         ax3.text(0.5, 0.5, 'Direction Matrix\n(Cross-correlation based)', 
#                 ha='center', va='center', transform=ax3.transAxes, fontsize=10)
#         ax3.axis('off')
    
#     # 4. Cross-correlation heatmap
#     ax4 = plt.subplot(2, 3, 4)
#     corr_matrix = results['cross_corr']
#     im4 = ax4.imshow(corr_matrix.values, cmap='RdBu_r', vmin=-1, vmax=1)
#     ax4.set_title('Cross-correlation Coefficient', fontsize=11, fontweight='bold')
#     ax4.set_xticks(range(len(corr_matrix.columns)))
#     ax4.set_yticks(range(len(corr_matrix.index)))
#     ax4.set_xticklabels(corr_matrix.columns, rotation=45, ha='right', fontsize=9)
#     ax4.set_yticklabels(corr_matrix.index, fontsize=9)
#     plt.colorbar(im4, ax=ax4, label='Correlation')
    
#     # # Add values for strong correlations
#     # for i in range(corr_matrix.shape[0]):
#     #     for j in range(corr_matrix.shape[1]):
#     #         if i != j and abs(corr_matrix.iloc[i, j]) > 0.5:
#     #             ax4.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
#     #                     ha='center', va='center', 
#     #                     color='white' if abs(corr_matrix.iloc[i, j]) > 0.7 else 'black',
#     #                     fontsize=7)
    
#     # 5. Time lags heatmap
#     ax5 = plt.subplot(2, 3, 5)
#     lag_matrix = results['cross_corr_lags']
#     im5 = ax5.imshow(lag_matrix.values, cmap='coolwarm', 
#                     vmin=-np.nanmax(np.abs(lag_matrix.values)), 
#                     vmax=np.nanmax(np.abs(lag_matrix.values)))
#     ax5.set_title('Optimal Time Lags (frames)', fontsize=11, fontweight='bold')
#     ax5.set_xticks(range(len(lag_matrix.columns)))
#     ax5.set_yticks(range(len(lag_matrix.index)))
#     ax5.set_xticklabels(lag_matrix.columns, rotation=45, ha='right', fontsize=9)
#     ax5.set_yticklabels(lag_matrix.index, fontsize=9)
#     plt.colorbar(im5, ax=ax5, label='Lag (frames)')
    
#     # Add lag values
#     # for i in range(lag_matrix.shape[0]):
#     #     for j in range(lag_matrix.shape[1]):
#     #         if i != j and lag_matrix.iloc[i, j] != 0:
#     #             color = 'white' if abs(lag_matrix.iloc[i, j]) > np.percentile(np.abs(lag_matrix.values), 75) else 'black'
#     #             ax5.text(j, i, f'{int(lag_matrix.iloc[i, j])}', 
#     #                     ha='center', va='center', color=color, fontsize=7)
    
#     # 6. Consensus matrix
#     ax6 = plt.subplot(2, 3, 6)
#     if not results['consensus'].empty:
#         consensus_matrix = results['consensus']
#         im6 = ax6.imshow(consensus_matrix.values, cmap='binary', vmin=0, vmax=1)
#         ax6.set_title('Consensus Connections\n(Multiple methods)', fontsize=11, fontweight='bold')
#         ax6.set_xticks(range(len(consensus_matrix.columns)))
#         ax6.set_yticks(range(len(consensus_matrix.index)))
#         ax6.set_xticklabels(consensus_matrix.columns, rotation=45, ha='right', fontsize=9)
#         ax6.set_yticklabels(consensus_matrix.index, fontsize=9)
        
#         # Count and display number of consensus connections
#         n_connections = np.sum(consensus_matrix.values) - len(consensus_matrix)  # Subtract diagonal
#         ax6.text(0.5, -0.15, f'Total: {int(n_connections)} connections', 
#                 transform=ax6.transAxes, ha='center', va='center', fontsize=9)
        
#         # Mark consensus connections
#         for i in range(consensus_matrix.shape[0]):
#             for j in range(consensus_matrix.shape[1]):
#                 if i != j and consensus_matrix.iloc[i, j] == 1:
#                     ax6.text(j, i, '✓', ha='center', va='center', 
#                             color='green', fontsize=10, fontweight='bold')
#     else:
#         ax6.text(0.5, 0.5, 'Consensus Matrix\n(Not enough methods)', 
#                 ha='center', va='center', transform=ax6.transAxes, fontsize=10)
#         ax6.axis('off')
    
#     plt.suptitle('Information Flow Analysis Results', fontsize=16, fontweight='bold', y=1.02)
#     plt.tight_layout()
#     plt.show()

# #%%
# # Example usage function
# # def run_analysis_for_condition(df_mean_ls, conditions, condition_idx, 
# #                               hemisphere='both', regions_of_interest=None):
# #     """
# #     Run information flow analysis for a specific condition.
    
# #     Parameters:
# #     -----------
# #     condition_idx : int
# #         Index of condition in conditions list (0=Hit, 1=Miss, 2=FA, 3=CR)
# #     """
#     # Extract time series for the specified condition
    
# hemisphere='both'
# regions_of_interest=None
# conditions = type_list
# all_ts_dict, region_names = extract_all_regions_timeseries(
#     df_mean_ls, conditions)

# # Filter regions if specified
# if regions_of_interest:
#     region_names = [r for r in region_names if r in regions_of_interest]

# # Create single condition time series dict
# condition_idx = 0
# condition = conditions[condition_idx]
# time_series_dict = {}

# for region in region_names:
#     if region in all_ts_dict and condition in all_ts_dict[region]:
#         time_series_dict[region] = all_ts_dict[region][condition]

# if not time_series_dict:
#     print(f"No valid time series found for condition {condition}")
#     # return None

# print(f"\n{'='*60}")
# print(f"ANALYZING CONDITION: {condition}")
# print(f"Regions: {len(time_series_dict)}")
# print(f"{'='*60}")

# # Run analysis pipeline
# results = analyze_information_flow_pipeline(time_series_dict, fs=10)

# # Visualize results
# visualize_all_results(results)

# # Plot network
# plot_causality_network(results['granger_causality'], threshold=0.05)
    

# #%%

# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import networkx as nx
# from matplotlib.colors import ListedColormap

# def cross_correlation_lag_analysis(time_series_dict, max_lag=10):
#     """
#     Find time lags between regions using cross-correlation.
#     """
#     regions = list(time_series_dict.keys())
#     n_regions = len(regions)
    
#     lag_matrix = np.zeros((n_regions, n_regions))
#     corr_matrix = np.zeros((n_regions, n_regions))
    
#     for i, region_i in enumerate(regions):
#         ts_i = np.array(time_series_dict[region_i])
#         mean_i = np.mean(ts_i)
#         ts_i_centered = ts_i - mean_i
        
#         for j, region_j in enumerate(regions):
#             if i != j:
#                 ts_j = np.array(time_series_dict[region_j])
#                 mean_j = np.mean(ts_j)
#                 ts_j_centered = ts_j - mean_j
                
#                 correlation = np.correlate(ts_i_centered, ts_j_centered, mode='full')
                
#                 norm = np.sqrt(np.sum(ts_i_centered**2) * np.sum(ts_j_centered**2))
#                 if norm > 0:
#                     correlation = correlation / norm
                
#                 n = len(ts_i)
#                 valid_lags = np.arange(-min(max_lag, n-1), min(max_lag, n-1)+1)
#                 valid_corr = correlation[n-1-min(max_lag, n-1):n+max_lag]
                
#                 peak_idx = np.argmax(np.abs(valid_corr))
#                 lag_matrix[i, j] = valid_lags[peak_idx]
#                 corr_matrix[i, j] = valid_corr[peak_idx]
    
#     lag_matrix[lag_matrix == max_lag] = np.nan
#     lag_matrix[lag_matrix == -max_lag] = np.nan
    
#     lag_df = pd.DataFrame(lag_matrix, index=regions, columns=regions)
#     corr_df = pd.DataFrame(corr_matrix, index=regions, columns=regions)
    
#     direction_df = pd.DataFrame('', index=regions, columns=regions)
#     for i in range(n_regions):
#         for j in range(n_regions):
#             if i != j:
#                 if lag_df.iloc[i, j] > 0:
#                     direction_df.iloc[i, j] = f"{regions[i]} → {regions[j]}"
#                 elif lag_df.iloc[i, j] < 0:
#                     direction_df.iloc[i, j] = f"{regions[j]} → {regions[i]}"
#                 else:
#                     direction_df.iloc[i, j] = "No lag"
    
#     return lag_df, corr_df, direction_df

# def extract_hemisphere_timeseries(df_mean_ls, conditions, region, hemisphere='both'):
#     """
#     Extract time series for a specific region and hemisphere.
#     """
#     time_series_dict = {}
    
#     for df, cond in zip(df_mean_ls, conditions):
#         if hemisphere == 'both':
#             left_col = f"{region}_l"
#             right_col = f"{region}_r"
            
#             if left_col in df.columns and right_col in df.columns:
#                 ts = (df[left_col] + df[right_col]) / 2
#             elif left_col in df.columns:
#                 ts = df[left_col]
#             elif right_col in df.columns:
#                 ts = df[right_col]
#             else:
#                 raise ValueError(f"Region {region} not found in DataFrame")
                
#         elif hemisphere in ['left', 'right']:
#             col_name = f"{region}_{hemisphere[0]}"
#             if col_name in df.columns:
#                 ts = df[col_name]
#             else:
#                 raise ValueError(f"Region {col_name} not found")
#         else:
#             raise ValueError("hemisphere must be 'left', 'right', or 'both'")
        
#         time_series_dict[cond] = ts.values
    
#     return time_series_dict

# def extract_all_hemispheres_timeseries(df_mean_ls, conditions):
#     """
#     Extract time series for ALL regions with separate hemispheres.
#     """
#     all_columns = df_mean_ls[0].columns
#     region_names = sorted(list(set(col[:-2] for col in all_columns 
#                                  if col.endswith(('_l', '_r')))))
    
#     results = {'left': {}, 'right': {}, 'both': {}}
    
#     for hemisphere in ['left', 'right', 'both']:
#         hemisphere_dict = {}
#         for region in region_names:
#             try:
#                 hemisphere_dict[region] = extract_hemisphere_timeseries(
#                     df_mean_ls, conditions, region, hemisphere
#                 )
#             except ValueError:
#                 continue
#         results[hemisphere] = hemisphere_dict
    
#     return results, region_names

# def plot_cross_correlation_results(corr_df, lag_df, direction_df, title):
#     """
#     Plot cross-correlation results for one condition.
#     """
#     fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
#     # 1. Correlation matrix
#     ax1 = axes[0]
#     im1 = ax1.imshow(corr_df.values, cmap='RdBu_r', vmin=-1, vmax=1)
#     ax1.set_title(f'{title}\nCross-correlation', fontsize=12, fontweight='bold')
#     ax1.set_xticks(range(len(corr_df.columns)))
#     ax1.set_yticks(range(len(corr_df.index)))
#     ax1.set_xticklabels(corr_df.columns, rotation=45, ha='right', fontsize=9)
#     ax1.set_yticklabels(corr_df.index, fontsize=9)
#     plt.colorbar(im1, ax=ax1, label='Correlation')
    
#     # 2. Lag matrix
#     ax2 = axes[1]
#     lag_vals = lag_df.values.copy()
#     lag_max = np.nanmax(np.abs(lag_vals))
#     im2 = ax2.imshow(lag_vals, cmap='coolwarm', vmin=-lag_max, vmax=lag_max)
#     ax2.set_title(f'{title}\nOptimal Time Lags (frames)', fontsize=12, fontweight='bold')
#     ax2.set_xticks(range(len(lag_df.columns)))
#     ax2.set_yticks(range(len(lag_df.index)))
#     ax2.set_xticklabels(lag_df.columns, rotation=45, ha='right', fontsize=9)
#     ax2.set_yticklabels(lag_df.index, fontsize=9)
#     plt.colorbar(im2, ax=ax2, label='Lag (frames)')
    
#     # 3. Direction matrix
#     ax3 = axes[2]
#     dir_vals = np.zeros(direction_df.shape)
#     regions = list(direction_df.index)
    
#     for i in range(len(regions)):
#         for j in range(len(regions)):
#             if i != j:
#                 cell_val = direction_df.iloc[i, j]
#                 if '→' in str(cell_val):
#                     dir_parts = str(cell_val).split(' → ')
#                     if len(dir_parts) == 2:
#                         if dir_parts[0] == regions[i]:
#                             dir_vals[i, j] = 1  # i→j
#                         else:
#                             dir_vals[i, j] = -1  # j→i
    
#     dir_cmap = ListedColormap(['red', 'gray', 'blue'])
#     im3 = ax3.imshow(dir_vals, cmap=dir_cmap, vmin=-1, vmax=1)
#     ax3.set_title(f'{title}\nInformation Flow Direction', fontsize=12, fontweight='bold')
#     ax3.set_xticks(range(len(direction_df.columns)))
#     ax3.set_yticks(range(len(direction_df.index)))
#     ax3.set_xticklabels(direction_df.columns, rotation=45, ha='right', fontsize=9)
#     ax3.set_yticklabels(direction_df.index, fontsize=9)
    
#     cbar = plt.colorbar(im3, ax=ax3, ticks=[-1, 0, 1])
#     cbar.ax.set_yticklabels(['j→i', 'No flow', 'i→j'])
    
#     for i in range(len(regions)):
#         for j in range(len(regions)):
#             if i != j and dir_vals[i, j] != 0:
#                 if dir_vals[i, j] == 1:
#                     arrow_char = '→'
#                     color = 'white'
#                 else:
#                     arrow_char = '←'
#                     color = 'black'
#                 ax3.text(j, i, arrow_char, ha='center', va='center', 
#                         color=color, fontsize=10, fontweight='bold')
    
#     plt.tight_layout()
#     plt.show()

# def analyze_cross_correlation_by_hemisphere(df_mean_ls, conditions, hemisphere='both'):
#     """
#     Run cross-correlation analysis for specific hemisphere.
#     """
#     all_ts_dict, region_names = extract_all_hemispheres_timeseries(df_mean_ls, conditions)
    
#     if hemisphere not in all_ts_dict:
#         raise ValueError(f"Hemisphere {hemisphere} not found in data")
    
#     hemisphere_data = all_ts_dict[hemisphere]
    
#     results_by_condition = {}
#     for cond in conditions:
#         time_series_dict = {}
#         for region in region_names:
#             if region in hemisphere_data and cond in hemisphere_data[region]:
#                 time_series_dict[region] = hemisphere_data[region][cond]
        
#         if time_series_dict:
#             print(f"\nAnalyzing {cond} (hemisphere: {hemisphere})")
#             lag_df, corr_df, direction_df = cross_correlation_lag_analysis(time_series_dict)
#             results_by_condition[cond] = {
#                 'lag': lag_df,
#                 'correlation': corr_df,
#                 'direction': direction_df
#             }
            
#             plot_cross_correlation_results(corr_df, lag_df, direction_df, 
#                                           f"{cond} - {hemisphere.capitalize()} Hemisphere")
    
#     return results_by_condition

# # def compare_hemispheres_single_condition(df_mean_ls, conditions, condition_idx=0):
# """
# Compare left vs right vs both hemispheres for one condition.
# """
# conditions = type_list
# condition_idx=0
# condition = conditions[condition_idx]
# print(f"\n{'='*60}")
# print(f"COMPARING HEMISPHERES FOR CONDITION: {condition}")
# print(f"{'='*60}")

# fig, axes = plt.subplots(3, 3, figsize=(15, 12))

# for idx, hemisphere in enumerate(['left', 'right', 'both']):
#     all_ts_dict, region_names = extract_all_hemispheres_timeseries(df_mean_ls, conditions)
#     hemisphere_data = all_ts_dict[hemisphere]
    
#     time_series_dict = {}
#     for region in region_names:
#         if region in hemisphere_data and condition in hemisphere_data[region]:
#             time_series_dict[region] = hemisphere_data[region][condition]
    
#     if not time_series_dict:
#         continue
    
#     lag_df, corr_df, direction_df = cross_correlation_lag_analysis(time_series_dict)
    
#     # Correlation matrix
#     ax1 = axes[idx, 0]
#     im1 = ax1.imshow(corr_df.values, cmap='RdBu_r', vmin=-1, vmax=1)
#     if idx == 0:
#         ax1.set_title('Correlation', fontsize=11, fontweight='bold')
#     ax1.set_ylabel(f'{hemisphere.capitalize()}', fontsize=10)
#     ax1.set_xticks([])
#     ax1.set_yticks([])
    
#     # Lag matrix
#     ax2 = axes[idx, 1]
#     lag_vals = lag_df.values.copy()
#     lag_max = np.nanmax(np.abs(lag_vals))
#     im2 = ax2.imshow(lag_vals, cmap='coolwarm', vmin=-lag_max, vmax=lag_max)
#     if idx == 0:
#         ax2.set_title('Time Lags', fontsize=11, fontweight='bold')
#     ax2.set_xticks([])
#     ax2.set_yticks([])
    
#     # Direction matrix
#     ax3 = axes[idx, 2]
#     dir_vals = np.zeros(direction_df.shape)
#     regions = list(direction_df.index)
    
#     for i in range(len(regions)):
#         for j in range(len(regions)):
#             if i != j:
#                 cell_val = direction_df.iloc[i, j]
#                 if '→' in str(cell_val):
#                     dir_parts = str(cell_val).split(' → ')
#                     if len(dir_parts) == 2:
#                         if dir_parts[0] == regions[i]:
#                             dir_vals[i, j] = 1
#                         else:
#                             dir_vals[i, j] = -1
    
#     dir_cmap = ListedColormap(['red', 'gray', 'blue'])
#     im3 = ax3.imshow(dir_vals, cmap=dir_cmap, vmin=-1, vmax=1)
#     if idx == 0:
#         ax3.set_title('Flow Direction', fontsize=11, fontweight='bold')
#     ax3.set_xticks([])
#     ax3.set_yticks([])

# plt.suptitle(f'Cross-correlation Analysis: {condition}', fontsize=14, fontweight='bold', y=1.02)
# plt.tight_layout()
# plt.show()

# # def run_cross_correlation_analysis(df_mean_ls, conditions):
# """
# Main function to run cross-correlation analysis.
# """
# print("=" * 60)
# print("CROSS-CORRELATION ANALYSIS")
# print("=" * 60)

# results = {}

# # Analyze each hemisphere separately
# for hemisphere in ['left', 'right', 'both']:
#     print(f"\nAnalyzing {hemisphere} hemisphere...")
#     hemisphere_results = analyze_cross_correlation_by_hemisphere(
#         df_mean_ls, conditions, hemisphere
#     )
#     results[hemisphere] = hemisphere_results

# # Compare hemispheres for each condition
# for i, condition in enumerate(conditions):
#     compare_hemispheres_single_condition(df_mean_ls, conditions, i)
    
#     # return results

# # # Example usage
# # if __name__ == "__main__":
# #     conditions = ['Hit', 'Miss', 'FA', 'CR']
    
# #     results = run_cross_correlation_analysis(df_mean_ls, conditions)
    
# #     # Optional: Save results
# #     import pickle
# #     with open('cross_correlation_results.pkl', 'wb') as f:
# #         pickle.dump(results, f)
# #     print("\nResults saved to 'cross_correlation_results.pkl'")

#%%

    



