# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from scipy.io import loadmat
import json

# Load images data
camId = loadmat('cuhk03_new_protocol_config_labeled.mat')['camId'].flatten()
filelist = loadmat('cuhk03_new_protocol_config_labeled.mat')['filelist'].flatten()
labels = loadmat('cuhk03_new_protocol_config_labeled.mat')['labels'].flatten()
gallery_idx, = loadmat('cuhk03_new_protocol_config_labeled.mat')['gallery_idx,'].flatten()
query_idx = loadmat('cuhk03_new_protocol_config_labeled.mat')['query_idx'].flatten()
train_idx = loadmat('cuhk03_new_protocol_config_labeled.mat')['train_idx'].flatten()

# Load features data
with open('feature_data.json', 'r') as f:
    features = json.load(f)
