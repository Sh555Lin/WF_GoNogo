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


base_dir = "/Volumes/Data_attention/Transfer learning/LinShu/DATA_linshu/000 Widefield"
mouse_id = "A095"
file_alignment = os.path.join(base_dir, mouse_id, 'widefield_alignment/wf_alignment_A095.pkl')
# Read a pickle file
with open(file_alignment, 'rb') as f:  # 'rb' means read binary
    data = pickle.load(f)


align = data['alignments']['20250815']