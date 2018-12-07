# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.cluster import KMeans

import json
import numpy as np
import matplotlib.pyplot as pd

# Load images data
camId = np.array(loadmat('cuhk03_new_protocol_config_labeled.mat')['camId'].flatten())
filelist = np.array(loadmat('cuhk03_new_protocol_config_labeled.mat')['filelist'].flatten())
labels = np.array(loadmat('cuhk03_new_protocol_config_labeled.mat')['labels'].flatten())
gallery_idx = np.array(loadmat('cuhk03_new_protocol_config_labeled.mat')['gallery_idx'].flatten())
query_idx = np.array(loadmat('cuhk03_new_protocol_config_labeled.mat')['query_idx'].flatten())
train_idx = np.array(loadmat('cuhk03_new_protocol_config_labeled.mat')['train_idx'].flatten())
data_size = camId.shape[0]

# Load features data
with open('feature_data.json', 'r') as f:
    features = np.array(json.load(f))

# Data preparation
train_idx = train_idx - 1
query_idx = query_idx - 1
gallery_idx = gallery_idx - 1

train_size = train_idx.shape[0]
query_size = query_idx.shape[0]
gallery_size = gallery_idx.shape[0]

train_cam = camId[train_idx]
query_cam = camId[query_idx]
gallery_cam = camId[gallery_idx]

train_label = labels[train_idx]
query_label = labels[query_idx]
gallery_label = labels[gallery_idx]

train_data = features[train_idx,:]
query_data = features[query_idx,:]
gallery_data = features[gallery_idx,:]

"""
score = 0
for q in range(query_size):
    cam_idx = [ idx for idx in range(gallery_size) if not( gallery_cam[idx] == query_cam[q] and gallery_label[idx] == query_label[q] )]    
    gallery_data_cam = gallery_data[cam_idx,:]
    gallery_label_cam = gallery_label[cam_idx]
    l2 = np.linalg.norm( (gallery_data_cam - query_data[q,:] ), ord=2, axis = 1)
    k_idx = np.argsort(l2)[:1]
    if gallery_label_cam[k_idx] == query_label[q]:
        score = score + 1
#print("success rate = " + str(mark/1400)
"""

kmeans = KMeans(n_clusters=700, random_state=300).fit(gallery_data)
cluster_idx = kmeans.labels_
#for k in range(700):
cluster_label_idx = [idx for idx in range(gallery_size) if cluster_idx[idx] == 0]
cluster_labels = gallery_label[cluster_label_idx]
count = np.bincount(cluster_labels).argmax()
    
#predict_kmeans = kmeans.predict(query_data)

#for q in range(query_size):
    