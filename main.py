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
import time
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
print("starting validation started...")
rank = 10
itr = [1, 3, 5]
val_knn_score = np.zeros((len(itr), val_num))
for v in range(val_num):
    # Non replace selection of validation set
    val_test_class= np.random.choice(label_trunc, 100, replace = False)
    trunc_delete_idx = [ np.argwhere(label_trunc == t) for t in val_test_class ]
    label_trunc = np.delete(label_trunc, trunc_delete_idx)
    # Extract data and label for each subset
    train_delete_idx =  [ i for t in val_test_class for i in range(train_size) if train_label[i] == t ]
    val_train_label = np.delete(train_label, train_delete_idx)
    val_train_data = np.delete(train_data, train_delete_idx, axis = 0)
    val_test_label = train_label[train_delete_idx]
    val_test_cam = train_cam[train_delete_idx]
    val_test_data = train_data[train_delete_idx,:]
    # Spit query and gallery sets
    query_rand_idx = [ i for t in val_test_class for i in np.random.choice( np.argwhere(val_test_label == t).flatten(), 2, replace = False ) ]
    val_query_label = val_test_label[query_rand_idx]
    val_query_data = val_test_data[query_rand_idx,:]
    val_query_size = val_query_label.shape[0]
    val_query_cam = val_test_cam[query_rand_idx]
    
    val_gallery_label = np.delete(val_test_label, query_rand_idx)
    val_gallery_data = np.delete(val_test_data, query_rand_idx, axis = 0)
    val_gallery_size = val_gallery_label.shape[0]
    val_gallery_cam = np.delete(val_test_cam, query_rand_idx)
    print("Validation ", v, " finished!")

    # PCA
    M = 35
    A_train, e_vals, e_vecs = compute_eigenspace(val_train_data, "high")
    idx=np.argsort(np.absolute(e_vals))[::-1]
    e_vals = e_vals[idx]
    e_vecs = (e_vecs.T[idx]).T
    m_vecs = e_vecs[:,0:M]
    train_pca = np.dot(val_train_data, m_vecs)
    
    '''
    N, D = query_data.shape
    X_avg = query_data.mean(0)
    X_avgm = np.array([X_avg]*N)
    A_query = (query_data - X_avgm).T'''
    query_pca = np.dot(val_query_data, m_vecs)
    '''
    N, D = gallery_data.shape
    X_avg = gallery_data.mean(0)
    X_avgm = np.array([X_avg]*N)
    A_gallery = (gallery_data - X_avgm).T'''
    gallery_pca = np.dot(val_gallery_data, m_vecs)
    
    for r in range(len(itr)):
        # Mahalonobis e_vals,
        #print("Starting Mahalonobis")
        maha_sup = mmc.MMC_Supervised(max_iter = itr[r], num_constraints = 20)
        #print("Starting fitting")
        maha_sup.fit(train_pca, val_train_label)
        A = maha_sup.metric()
        query_trans = maha_sup.transform(query_pca)
        gallery_trans = maha_sup.transform(gallery_pca)
        
        #Kernel
        
        # Classification
        #print("Starting classification")
        knn_score = knn_classifier(rank, val_query_size, query_trans, val_query_label, val_query_cam, val_gallery_size, gallery_trans, val_gallery_cam, val_gallery_label)
        #kmeans_score = kmeans_classifier(100, val_query_size, query_trans, val_query_label, val_gallery_size, gallery_trans, val_gallery_label )
        #score.append(knn_score)
        val_knn_score[r,v] = knn_score
        print("Fuckin done")

val_score = np.mean(val_knn_score, axis = 1)
opt_itr = itr[np.argmax(val_score)]

# PCA
A_train, e_vals, e_vecs = compute_eigenspace(train_data, "high")
idx=np.argsort(np.absolute(e_vals))[::-1]
e_vals = e_vals[idx]
e_vecs = (e_vecs.T[idx]).T
m_vecs = e_vecs[:,0:M]
train_pca = np.dot(train_data, m_vecs)

'''
N, D = query_data.shape
X_avg = query_data.mean(0)
X_avgm = np.array([X_avg]*N)
A_query = (query_data - X_avgm).T'''
query_pca = np.dot(query_data, m_vecs)
'''
N, D = gallery_data.shape
X_avg = gallery_data.mean(0)
X_avgm = np.array([X_avg]*N)
A_gallery = (gallery_data - X_avgm).T'''
gallery_pca = np.dot(gallery_data, m_vecs)
# Mahalonobis e_vals,
print("Starting Mahalonobis")
maha_sup = mmc.MMC_Supervised(max_iter = opt_itr, num_constraints = 20)
print("Starting fitting")
maha_sup.fit(train_pca, train_label)
print("Fuckin done")
A = maha_sup.metric()
query_trans = maha_sup.transform(query_pca)
gallery_trans = maha_sup.transform(gallery_pca)

#Kernel

# Classification
print("Starting classification")
test_knn_score = knn_classifier(rank, query_size, query_trans, query_label, query_cam, gallery_size, gallery_trans, gallery_cam, gallery_label)
#kmeans_score = kmeans_classifier(100, val_query_size, query_trans, val_query_label, val_gallery_size, gallery_trans, val_gallery_label )
#score.append(knn_score)

    
        
        
        
    