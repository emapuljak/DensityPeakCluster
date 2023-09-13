#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import gc
from sklearn.cluster import DBSCAN,HDBSCAN
import numpy as np
from distance.distance_builder import *
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import roc_curve, auc


def get_roc_data(qcd, bsm):
    true_val = np.concatenate((np.ones(bsm.shape[0]), np.zeros(qcd.shape[0])))
    pred_val = np.nan_to_num(np.concatenate((bsm, qcd)))
    fpr_loss, tpr_loss, threshold_loss = roc_curve(true_val, pred_val, drop_intermediate=False)
    auc_data = auc(fpr_loss, tpr_loss)
    return fpr_loss, tpr_loss, auc_data

if __name__ == '__main__':

    # builder = DistanceBuilder()
    # builder.load_points(r'data/data_jets/jets_lat8.data')

    # latent space vectors
    # input_vectors = builder.vectors

    # distances between latent space vectors
    dist_matrix = np.zeros((6000, 6000))
    max_id = 6000
    with open('data/data_jets/jets_lat8_l1dist.forcluster', 'r') as fp:
        for line in fp:
            x1, x2, d = line.strip().split(' ')
            x1, x2 = int(x1), int(x2)
            dist_matrix[x1-1, x2-1] = float(d)
            dist_matrix[x2-1, x1-1] = float(d)
    for i in range(max_id):
        dist_matrix[i, i] = 0.0

    ####### clustering vectors ######
    #clustering = DBSCAN(eps=0.2, min_samples=100).fit(input_vectors)
    clustering = DBSCAN(eps=0.39, min_samples=50, metric='precomputed').fit(dist_matrix)
    labels = clustering.labels_

    print(np.unique(labels, return_counts=True))

    core_samples = np.zeros_like(labels, dtype = bool)
    core_samples[clustering.core_sample_indices_] = True

    print(f'Core: {core_samples}')

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
    print("Estimated number of clusters: %d" % n_clusters_)
    print("Estimated number of noise points: %d" % n_noise_)

    del dist_matrix
    gc.collect()

    # builder = DistanceBuilder()
    # builder.load_points(r'data/data_jets/jets_lat8.data')

    # # latent space vectors
    # input_vectors = builder.vectors


    # # plot results
    # df = pd.DataFrame(input_vectors)
    # df['class_id'] = labels
    # figure = sns.pairplot(df, hue='class_id', diag_kind="hist", palette="bright")
    # plt.tight_layout()
    # plt.show()

    # del input_vectors
    # gc.collect()

    # distances for test QCD
    dist_matrix_qcd = np.zeros((10000, 10000))
    max_id = 10000
    with open('data/data_jets/jets_test_lat8_l1dist.forcluster', 'r') as fp:
        for line in fp:
            x1, x2, d = line.strip().split(' ')
            x1, x2 = int(x1), int(x2)
            dist_matrix_qcd[x1-1, x2-1] = float(d)
            dist_matrix_qcd[x2-1, x1-1] = float(d)
    for i in range(max_id):
        dist_matrix_qcd[i, i] = 0.0

    # distances for test SIG
    dist_matrix_sig = np.zeros((10000, 10000))
    max_id = 10000
    with open('data/data_jets/jets_NA35Graviton_lat8_l1dist.forcluster', 'r') as fp:
        for line in fp:
            x1, x2, d = line.strip().split(' ')
            x1, x2 = int(x1), int(x2)
            dist_matrix_sig[x1-1, x2-1] = float(d)
            dist_matrix_sig[x2-1, x1-1] = float(d)
    for i in range(max_id):
        dist_matrix_sig[i, i] = 0.0

    # load evaluation data
    # builder = DistanceBuilder()
    # builder.load_points(r'data/data_jets/jets_test_lat8.data')
    # input_vectors_qcd = builder.vectors

    # builder = DistanceBuilder()
    # builder.load_points(r'data/data_jets/jets_NA35Graviton_lat8.data')
    # input_vectors_sig = builder.vectors

    # label evaluation data
    labels_qcd = clustering.fit(dist_matrix_qcd).labels_
    labels_qcd = np.array([1 if label==-1 else label for label in labels_qcd])
    print('Test QCD')
    print(np.unique(labels_qcd))
    labels_sig = clustering.fit(dist_matrix_sig).labels_
    labels_sig = np.array([1 if label==-1 else label for label in labels_sig])
    print('Test SIG')
    print(np.unique(labels_sig))
    roc_data = get_roc_data(labels_qcd, labels_sig)
    x = roc_data[1]; y = roc_data[0]

    fig = plt.figure(figsize=(8,8))
    plt.plot(x, y, label='(auc = %.2f)'% (roc_data[2]*100.), linewidth=1.5)
    plt.ylabel('FPR')
    plt.xlabel('TPR')
    plt.yscale('log')
    plt.grid(True)
    plt.legend(fancybox=True, frameon=True, prop={"size":10}, bbox_to_anchor =(1.0, 1.0))
    plt.show()
    
