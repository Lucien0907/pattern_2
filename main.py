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


label_trunc = np.unique(train_label)
val_num = 7
for v in range(val_num):
    print(v)
    val_test_label= np.random.choice(label_trunc, 100, replace = False)
    trunc_delete_idx = [ np.argwhere(label_trunc == t) for t in val_test_label ]
    label_trunc = np.delete(label_trunc, trunc_delete_idx)
    
    train_delete_idx =  [ i for t in val_test_label for i in range(train_size) if train_label[i] == t ]
    val_label = train_label[train_delete_idx]
    val_test_data = train_data[val_test_label,:]      
    val_train_label = np.delete(train_label, train_delete_idx)
    val_train_data = np.delete(train_data, train_delete_idx, axis = 0)

'''
# PCA
M = 35
A_train, e_vals, e_vecs = compute_eigenspace(train_data, "high")
idx=np.argsort(np.absolute(e_vals))[::-1]
e_vals = e_vals[idx]
e_vecs = (e_vecs.T[idx]).T
m_vecs = e_vecs[:,0:M]
train_pca = np.dot(train_data, m_vecs)
'''
'''

N, D = query_data.shape
X_avg = query_dataa numpy array, and there's a very simple metho.mean(0)
X_avgm = np.array([X_avg]*N)
A_query = (query_data - X_avgm).T
query_pca = np.dot(query_data, m_vecs)
'''
'''
N, D = gallery_data.shape
X_avg = gallery_data.mean(0)
X_avgm = np.array([X_avg]*N)
A_gallery = (gallery_data - X_avgm).T
gallery_pca = np.dot(query_data, m_vecs)

'''
'''
A_train, e_vals_train, e_vecs_train = compute_eigenspace(train_data, "high")
idx=np.argsort(np.absolute(e_vals_train))[::-1]
e_vals_train = e_vals_train[idx]
e_vecs_train = (e_vecs_train.T[idx]).T
m_vecs_train = e_vecs_train[:,0:M]
train_pca = np.dot(A_train.T, m_vecs_train)

A_query, e_vals_query, e_vecs_query = compute_eigenspace(query_data, "high")
idx=np.argsort(np.absolute(e_vals_query))[::-1]
e_vals_query = e_vals_query[idx]
e_vecs_query = (e_vecs_query.T[idx]).T
m_vecs_query = e_vecs_query[:,0:M]
query_pca = np.dot(A_query.T, m_vecs_query)np.unique(a)np.unique(a)
array([1, 2, 3])
array([1, 2, 3])

A_gallery, e_vals_gallery, e_vecs_gallery = compute_eigenspace(gallery_data, "high")
idx=np.argsort(np.absolute(e_vals_gallery))[::-1]
e_vals_gallery = e_vals_gallery[idx]
e_vecs_gallery = (e_vecs_gallery.T[idx]).T
m_vecs_gallery = e_vecs_gallery[:,0:M]
gallery_pca = np.dot(A_gallery.T, m_vecs_gallery)
'''
'''
# Mahalonobis e_vals,
print("Starting Mahalonobis")
maha_sup = mmc.MMC_Supervised(max_iter = 100, num_constraints = 20)
print("Starting fitting")
maha_sup.fit(train_pca, train_label)
print("Fuckin done")
A = maha_sup.metric()
query_trans = maha_sup.transform(query_pca)
gallery_trans = maha_sup.transform(gallery_pca)
'''
#Kernel

'''
# Rename for base line
query_trans = query_data
gallery_trans = gallery_data
# Classification
print("Starting classification")
knn_score = knn_classifier(query_size, query_trans, query_label,query_cam, gallery_size, gallery_trans, gallery_cam, gallery_label)
kmeans_score = kmeans_classifier(class_size, query_size, query_trans, query_label, gallery_size, gallery_trans, gallery_label )
'''

