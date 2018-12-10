# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier  
from sklearn.cluster import KMeans
from sklearn.neighbors import DistanceMetric
from scipy.spatial import distance

import json
from metric_learn import mmc
import numpy as np
from pr_function import knn_classifier, kmeans_classifier, compute_eigenspace

# Load images data
camId = np.array(loadmat('cuhk03_new_protocol_config_labeled.mat')['camId'].flatten())
filelist = np.array(loadmat('cuhk03_new_protocol_config_labeled.mat')['filelist'].flatten())
labels = np.array(loadmat('cuhk03_new_protocol_config_labeled.mat')['labels'].flatten())
gallery_idx = np.array(loadmat('cuhk03_new_protocol_config_labeled.mat')['gallery_idx'].flatten())
query_idx = np.array(loadmat('cuhk03_new_protocol_config_labeled.mat')['query_idx'].flatten())
train_idx = np.array(loadmat('cuhk03_new_protocol_config_labeled.mat')['train_idx'].flatten())
data_size = camId.shape[0]
class_size = 700

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

# PCA
X_avg, A, e_vals, e_vecs = compute_eigenspace(train_data, "high")

# Sort eigen vectors and eigen value in descending order
idx=np.argsort(np.absolute(e_vals))[::-1]
e_vals = e_vals[idx]
e_vecs = (e_vecs.T[idx]).T

m_vecs = e_vecs[:,0:50]
train_pca = np.dot(train_data, m_vecs)
query_pca = np.dot(query_data, m_vecs)
gallery_pca = np.dot(gallery_data, m_vecs)

# Mahalonobis
print("Starting Mahalonobis")
maha_sup = mmc.MMC_Supervised(max_iter = 100, num_constraints = 1)
print("Starting fitting")
maha_sup.fit(train_pca, train_label)
print("Fuckin done")
A = maha_sup.metric()
query_trans = maha_sup.transform(query_pca)
gallery_trans = maha_sup.transform(gallery_pca)

#Kernel


# Rename for base line
#query_trans = query_data
#gallery_trans = gallery_data
# Classification
print("Starting classification")
knn_score = knn_classifier(query_size, query_trans, query_label,query_cam, gallery_size, gallery_trans, gallery_cam, gallery_label)
kmeans_score = kmeans_classifier(class_size, query_size, query_trans, query_label, gallery_size, gallery_trans, gallery_label )


