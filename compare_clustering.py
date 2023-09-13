import warnings
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from itertools import cycle, islice

from sklearn import cluster, datasets, mixture
from sklearn.neighbors import kneighbors_graph

from distance.distance_builder import *
from utils import *

def load_data(path, n_samples=512):
    builder = DistanceBuilder()
    builder.load_points(path)
    # latent space vectors
    data = builder.vectors[:n_samples]
    return data

if __name__ == '__main__':
    # ============
    # Load data - latent space
    # ============

    X_train = load_data(r'data/data_jets/lat16/jets_lat16.data', n_samples=512)
    X_test = load_data(r'data/data_jets/lat16/jets_test_lat16.data', n_samples=20000)
    X_sig = load_data(r'data/data_jets/lat16/jets_NA35Graviton_lat16.data', n_samples=20000)

    # ============
    # Set up cluster parameters
    # ============

    params = {
        "batch_size": 64,
        "quantile": 0.3,
        "eps": 0.2,
        "damping": 0.9,
        "preference": -200,
        "n_neighbors": 3,
        "n_clusters": 2,
        "min_samples": 100,
        "xi": 0.05,
        "min_cluster_size": 0.1,
        "allow_single_cluster": True,
        "hdbscan_min_cluster_size": 15,
        "hdbscan_min_samples": 3,
    }

    # estimate bandwidth for mean shift
    bandwidth = cluster.estimate_bandwidth(X_train, quantile=params["quantile"])

    # connectivity matrix for structured Ward
    # connectivity = kneighbors_graph(
    #     X, n_neighbors=params["n_neighbors"], include_self=False
    # )
    # make connectivity symmetric
    # connectivity = 0.5 * (connectivity + connectivity.T)

    # ============
    # Create cluster objects
    # ============
    ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
    two_means_batch = cluster.MiniBatchKMeans(n_clusters=params["n_clusters"], n_init="auto", batch_size=params["batch_size"])
    two_means = cluster.KMeans(n_clusters=params["n_clusters"], n_init="auto")
    # ward = cluster.AgglomerativeClustering(
    #     n_clusters=params["n_clusters"], linkage="ward", connectivity=connectivity
    # )
    # spectral = cluster.SpectralClustering(
    #     n_clusters=params["n_clusters"],
    #     eigen_solver="arpack",
    #     affinity="nearest_neighbors",
    # )
    # dbscan = cluster.DBSCAN(eps=params["eps"], min_samples=params["min_samples"])
    hdbscan = cluster.HDBSCAN(
        min_samples=params["hdbscan_min_samples"],
        min_cluster_size=params["hdbscan_min_cluster_size"],
        cluster_selection_epsilon=params["eps"],
        allow_single_cluster=params["allow_single_cluster"],
        store_centers='centroid'
    )
    # optics = cluster.OPTICS(
    #     min_samples=params["min_samples"],
    #     xi=params["xi"],
    #     min_cluster_size=params["min_cluster_size"],
    # )
    affinity_propagation = cluster.AffinityPropagation(
        damping=params["damping"], preference=params["preference"], random_state=0
    )
    # average_linkage = cluster.AgglomerativeClustering(
    #     linkage="average",
    #     metric="cityblock",
    #     n_clusters=params["n_clusters"],
    #     connectivity=connectivity,
    # )
    # birch = cluster.Birch(n_clusters=params["n_clusters"]) - tree clustering
    # gmm = mixture.GaussianMixture(
    #     n_components=params["n_clusters"], covariance_type="full"
    # )

    clustering_algorithms = (
            ("MiniBatch KMeans", two_means_batch),
            ("KMeans", two_means),
            ("Affinity Propagation", affinity_propagation),
            ("MeanShift", ms),
            # ("Spectral Clustering", spectral),
            # ("Ward", ward),
            # ("Agglomerative Clustering", average_linkage),
            # ("DBSCAN", dbscan),
            ("HDBSCAN", hdbscan),
            # ("OPTICS", optics),
            # ("BIRCH", birch),
            # ("Gaussian Mixture", gmm),
        )

    for name, algorithm in clustering_algorithms:
        #t0 = time.time()

        # catch warnings related to kneighbors_graph
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="the number of connected components of the "
                + "connectivity matrix is [0-9]{1,2}"
                + " > 1. Completing it to avoid stopping the tree early.",
                category=UserWarning,
            )
            warnings.filterwarnings(
                "ignore",
                message="Graph is not fully connected, spectral embedding"
                + " may not work as expected.",
                category=UserWarning,
            )
            algorithm.fit(X_train)

        #t1 = time.time()

        # find centroids
        if name in ['MiniBatch KMeans', 'KMeans', 'Affinity Propagation', 'MeanShift']: centroids = algorithm.cluster_centers_
        elif name == 'HDBSCAN': centroids = algorithm.centroids_
        else: 
            ValueError('Wrong name of algorithm')

        # find AD score of test data
        # QCD
        dist_qcd = find_distances_to_centroids(X_test, centroids)
        print(dist_qcd.shape)
        score_qcd = ad_score(distances=dist_qcd)
        loss_qcd = combine_loss_min(score_qcd)

        # SIG
        dist_sig = find_distances_to_centroids(X_sig, centroids)
        print(dist_sig.shape)
        score_sig = ad_score(distances=dist_sig)
        loss_sig = combine_loss_min(score_sig)

        # find cluster assignments
        # if hasattr(algorithm, "labels_"):
        #     y_pred_qcd = algorithm.fit(X_test).labels_.astype(int)
        #     y_pred_sig = algorithm.fit(X_sig).labels_.astype(int)
        # else:
        #     y_pred_qcd = algorithm.predict(X_test)
        #     y_pred_sig = algorithm.predict(X_sig)
        
        # df = pd.DataFrame(input_vectors)
        # df['class_id'] = y_pred
        # figure = sns.pairplot(df, hue='class_id', diag_kind="hist", palette='bright')
        # figure.fig.suptitle(name, fontsize=70)
        # plt.tight_layout()
        # plt.show()

        roc_data = get_roc_data(loss_qcd, loss_sig)
        x = roc_data[1]; y = roc_data[0]

        fig = plt.figure(figsize=(8,8))
        plt.plot(x, y, label='(auc = %.2f)'% (roc_data[2]*100.), linewidth=1.5)
        plt.ylabel('FPR')
        plt.xlabel('TPR')
        plt.grid(True)
        plt.legend(fancybox=True, frameon=True, prop={"size":10}, bbox_to_anchor =(1.0, 1.0))
        plt.title(name, fontsize=25)
        plt.savefig(f'results/results_jets/lat16/{name}.png')


