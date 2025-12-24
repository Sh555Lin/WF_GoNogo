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
from tifffile import imwrite
import pathlib

abspath = os.path.abspath(__file__)
current_dir = os.path.dirname(abspath)
parent_dir = os.path.dirname(current_dir)
os.chdir(parent_dir)
if parent_dir not in sys.path: 
    sys.path.append(parent_dir)
from utils.wf_utils import *
# import pickle

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
    
base_dir = "/Volumes/Data_attention/Transfer learning/LinShu/DATA_linshu/000 Widefield"
mouse_id = "A095"
file_alignment = os.path.join(base_dir, mouse_id, '20250903/process/ccf_regions.pkl')#'widefield_alignment/wf_alignment_A095.pkl')
# Read a pickle file
with open(file_alignment, 'rb') as f:  # 'rb' means read binary
    data = load_pickle_safely(f)


#%% generate eight dataframes: (Average, single trial) X (Hit, Miss, FA, CR)



visp_data = data[data['acronym'] == 'VISp']
plt.plot(np.array(visp_data['left_x'].values[0]),np.array(visp_data['left_y'].values[0]),'.')



import matplotlib.cm as cm

unique_regions = data['acronym'].unique()
colors = cm.rainbow(np.linspace(0, 1, len(unique_regions)))

plt.figure(figsize=(12, 10))

for i, region in enumerate(unique_regions):
    region_data = data[data['acronym'] == region]
    
    if len(region_data) > 0:
        # Combine left and right coordinates
        x_coords = np.concatenate([
            np.array(region_data['left_x'].values[0]),
            np.array(region_data['right_x'].values[0])
        ])
        y_coords = np.concatenate([
            np.array(region_data['left_y'].values[0]),
            np.array(region_data['right_y'].values[0])
        ])
        
        plt.scatter(x_coords, y_coords, 
                   color=colors[i], label=region, alpha=0.7, s=50)

plt.xlabel('X')
plt.ylabel('Y')
plt.title('All Brain Regions')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


#%%
# Create a directory to save the masks
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from scipy import ndimage
from scipy.spatial import ConvexHull
import os

# Create directories to save masks
save_dir_left = "brain_region_masks_left"
save_dir_right = "brain_region_masks_right"
os.makedirs(save_dir_left, exist_ok=True)
os.makedirs(save_dir_right, exist_ok=True)

# Get all unique region acronyms
unique_regions = data['acronym'].unique()
colors = cm.rainbow(np.linspace(0, 1, len(unique_regions)))

# Determine matrix size from your data
max_x = 0
max_y = 0

# First pass: find maximum coordinates
for region in unique_regions:
    region_data = data[data['acronym'] == region]
    if len(region_data) > 0:
        # Check left side
        left_x = np.array(region_data['left_x'].values[0])
        left_y = np.array(region_data['left_y'].values[0])
        if len(left_x) > 0:
            max_x = max(max_x, np.max(left_x))
            max_y = max(max_y, np.max(left_y))
        
        # Check right side
        right_x = np.array(region_data['right_x'].values[0])
        right_y = np.array(region_data['right_y'].values[0])
        if len(right_x) > 0:
            max_x = max(max_x, np.max(right_x))
            max_y = max(max_y, np.max(right_y))

# Add padding
matrix_size = int(max(max_x, max_y)) + 50

# Dictionaries to store masks
left_region_masks = {}
right_region_masks = {}

plt.figure(figsize=(15, 6))

# Plot left regions
plt.subplot(1, 2, 1)
for i, region in enumerate(unique_regions):
    region_data = data[data['acronym'] == region]
    
    if len(region_data) > 0:
        # LEFT REGION
        left_x = np.array(region_data['left_x'].values[0])
        left_y = np.array(region_data['left_y'].values[0])
        
        if len(left_x) > 0:
            # Convert to integers
            left_x_int = left_x.astype(int)
            left_y_int = left_y.astype(int)
            
            # Create convex hull for the region
            if len(left_x_int) >= 3:
                try:
                    points = np.column_stack([left_x_int, left_y_int])
                    hull = ConvexHull(points)
                    
                    # Create mask and fill convex hull
                    left_mask = np.zeros((matrix_size, matrix_size), dtype=bool)
                    
                    # Get convex hull vertices
                    hull_vertices = points[hull.vertices]
                    
                    # Fill the convex hull polygon
                    from matplotlib.path import Path
                    grid_x, grid_y = np.meshgrid(np.arange(matrix_size), np.arange(matrix_size))
                    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
                    
                    path = Path(hull_vertices)
                    inside = path.contains_points(grid_points)
                    left_mask = inside.reshape((matrix_size, matrix_size)).T
                    
                except:
                    # Fallback: create mask from points and dilate
                    left_mask = np.zeros((matrix_size, matrix_size), dtype=bool)
                    valid_indices = (left_x_int >= 0) & (left_x_int < matrix_size) & \
                                   (left_y_int >= 0) & (left_y_int < matrix_size)
                    left_mask[left_x_int[valid_indices], left_y_int[valid_indices]] = True
                    
                    # Dilate to fill interior
                    left_mask = ndimage.binary_dilation(left_mask, structure=np.ones((3, 3)), iterations=5)
                    left_mask = ndimage.binary_fill_holes(left_mask)
            else:
                # For regions with few points, create dilated mask
                left_mask = np.zeros((matrix_size, matrix_size), dtype=bool)
                valid_indices = (left_x_int >= 0) & (left_x_int < matrix_size) & \
                               (left_y_int >= 0) & (left_y_int < matrix_size)
                left_mask[left_x_int[valid_indices], left_y_int[valid_indices]] = True
                left_mask = ndimage.binary_dilation(left_mask, structure=np.ones((5, 5)), iterations=3)
                left_mask = ndimage.binary_fill_holes(left_mask)
            
            # Store mask
            left_region_masks[region] = left_mask
            
            # Save mask
            np.save(os.path.join(save_dir_left, f"{region}_left_mask.npy"), left_mask)
            
            # Plot
            plt.scatter(left_x, left_y, color=colors[i], label=region, alpha=0.7, s=30)
            
            print(f"{region} LEFT: {left_mask.sum()} pixels filled")

plt.title('Left Brain Regions')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True, alpha=0.3)

# Plot right regions
plt.subplot(1, 2, 2)
for i, region in enumerate(unique_regions):
    region_data = data[data['acronym'] == region]
    
    if len(region_data) > 0:
        # RIGHT REGION
        right_x = np.array(region_data['right_x'].values[0])
        right_y = np.array(region_data['right_y'].values[0])
        
        if len(right_x) > 0:
            # Convert to integers
            right_x_int = right_x.astype(int)
            right_y_int = right_y.astype(int)
            
            # Create convex hull for the region
            if len(right_x_int) >= 3:
                try:
                    points = np.column_stack([right_x_int, right_y_int])
                    hull = ConvexHull(points)
                    
                    # Create mask and fill convex hull
                    right_mask = np.zeros((matrix_size, matrix_size), dtype=bool)
                    
                    # Get convex hull vertices
                    hull_vertices = points[hull.vertices]
                    
                    # Fill the convex hull polygon
                    from matplotlib.path import Path
                    grid_x, grid_y = np.meshgrid(np.arange(matrix_size), np.arange(matrix_size))
                    grid_points = np.column_stack([grid_x.ravel(), grid_y.ravel()])
                    
                    path = Path(hull_vertices)
                    inside = path.contains_points(grid_points)
                    right_mask = inside.reshape((matrix_size, matrix_size)).T
                    
                except:
                    # Fallback: create mask from points and dilate
                    right_mask = np.zeros((matrix_size, matrix_size), dtype=bool)
                    valid_indices = (right_x_int >= 0) & (right_x_int < matrix_size) & \
                                   (right_y_int >= 0) & (right_y_int < matrix_size)
                    right_mask[right_x_int[valid_indices], right_y_int[valid_indices]] = True
                    
                    # Dilate to fill interior
                    right_mask = ndimage.binary_dilation(right_mask, structure=np.ones((3, 3)), iterations=5)
                    right_mask = ndimage.binary_fill_holes(right_mask)
            else:
                # For regions with few points, create dilated mask
                right_mask = np.zeros((matrix_size, matrix_size), dtype=bool)
                valid_indices = (right_x_int >= 0) & (right_x_int < matrix_size) & \
                               (right_y_int >= 0) & (right_y_int < matrix_size)
                right_mask[right_x_int[valid_indices], right_y_int[valid_indices]] = True
                right_mask = ndimage.binary_dilation(right_mask, structure=np.ones((5, 5)), iterations=3)
                right_mask = ndimage.binary_fill_holes(right_mask)
            
            # Store mask
            right_region_masks[region] = right_mask
            
            # Save mask
            np.save(os.path.join(save_dir_right, f"{region}_right_mask.npy"), right_mask)
            
            # Plot
            plt.scatter(right_x, right_y, color=colors[i], label=region, alpha=0.7, s=30)

plt.title('Right Brain Regions')
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("separated_brain_regions.png", dpi=150, bbox_inches='tight')
plt.show()

# Create filled region visualizations
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

# Display some example filled regions
example_regions = list(unique_regions)[:6]

for i, region in enumerate(example_regions):
    ax = axes[i]
    
    # Combine left and right for visualization
    if region in left_region_masks and region in right_region_masks:
        combined_mask = left_region_masks[region] | right_region_masks[region]
    elif region in left_region_masks:
        combined_mask = left_region_masks[region]
    else:
        combined_mask = right_region_masks[region]
    
    ax.imshow(combined_mask, cmap='hot', interpolation='nearest')
    ax.set_title(f"{region}\n{combined_mask.sum()} pixels")
    ax.axis('off')

plt.tight_layout()
plt.savefig("filled_regions_examples.png", dpi=150, bbox_inches='tight')
plt.show()

# Save all masks in compressed format
np.savez_compressed(os.path.join(save_dir_left, "all_left_masks.npz"), **left_region_masks)
np.savez_compressed(os.path.join(save_dir_right, "all_right_masks.npz"), **right_region_masks)

# Summary
print("\n" + "="*50)
print("SUMMARY")
print("="*50)
print(f"Matrix size: {matrix_size}x{matrix_size}")
print(f"Total regions: {len(unique_regions)}")
print(f"Left masks created: {len(left_region_masks)}")
print(f"Right masks created: {len(right_region_masks)}")
print(f"\nMask directories:")
print(f"  Left masks: {save_dir_left}")
print(f"  Right masks: {save_dir_right}")