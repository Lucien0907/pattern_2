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
import metric_learn
import numpy as np

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

'''
# take 1:
score = 0
IV=np.linalg.inv(np.cov(train_data.T))
for q in range(query_size):
    dis_q = [ distance.mahalanobis(query_data[q,:], gallery_data[g,:], IV) for g in range(gallery_size) ]
    k_idx = np.argsort(dis_q)[:1]
    if gallery_label[k_idx] == query_label[q]:
        score = score + 1
    print(q)
print(score/query_size)  
'''

'''
# take 2: 
dis = DistanceMetric.get_metric( 'mahalanobis', V=np.cov(train_data.T) )
score = 0
for q in range(query_size):
    dis_q = [ dis.pairwise([query_data[q,:], gallery_data[g,:]])[0,1] for g in range(gallery_size) ]
    k_idx = np.argsort(dis_q)[:1]
    if gallery_label[k_idx] == query_label[q]:
        score = score + 1
    print(q)
print(score/query_size)  
'''

"""
# KNN
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

"""
# Kmeans
kmeans = KMeans(n_clusters=700, random_state=300).fit(gallery_data)
cluster_idx = kmeans.labels_
cluster_labels = np.zeros(700)
for k in range(700):
    cluster_label_idx = [idx for idx in range(gallery_size) if cluster_idx[idx] == k]
    cluster_all_labels = gallery_label[cluster_label_idx]
    label = np.bincount(cluster_all_labels).argmax()
    cluster_labels[k] = label
predict_kmeans = kmeans.predict(query_data)

score = 0
for q in range(query_size):
    if cluster_labels[predict_kmeans[q]] == query_label[q]:
        score = score + 1
"""

  
    #for g in range(gallery_size):
        #dis = distance.mahalanobis(query_data[q,:], gallery_data[g,:], IV)
        #if g == 0:
            #distant = dis
        #else:
            #distant = np.hstack((distant, dis))
