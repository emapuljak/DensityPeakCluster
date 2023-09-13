import numpy as np
from sklearn.metrics import roc_curve, auc

def find_distances_to_centroids(points, centroids):
    
    """
    Args:
        points: numpy.ndarray of shape (N, X)
                    N = number of samples,
                    X = dimension of latent space;
        centroids: numpy.ndarray of shape (N, X)
    Returns:
        cluster_assignments: numpy.ndarray of shape (N, X) specifying to which cluster each feature is assigned
        distances: numpy.ndarray of shape (N, X) specifying distances to nearest cluster
    """
    
    n = points.shape[0]
    k = centroids.shape[0] # number of centroids
    distances=[]
    
    for i in range(n): # through all training samples
        dist=[]
        for j in range(k): # distance of each training example to each centroid
            temp_dist = np.linalg.norm(points[i,:] - centroids[j,:]) # returning back one number for all latent dimensions!
            dist.append(temp_dist)
        distances.append(dist)
    return np.asarray(distances)

def ad_score(distances, method='sum_all'):
    if method=='sum_all':
        return np.sqrt(np.sum(distances**2, axis=1))
    
def combine_loss_min(loss):
    loss_j1, loss_j2 = np.split(loss, 2)
    return np.minimum(loss_j1, loss_j2)

def get_roc_data(qcd, bsm):
    true_val = np.concatenate((np.ones(bsm.shape[0]), np.zeros(qcd.shape[0])))
    pred_val = np.nan_to_num(np.concatenate((bsm, qcd)))
    fpr_loss, tpr_loss, threshold_loss = roc_curve(true_val, pred_val, drop_intermediate=False)
    auc_data = auc(fpr_loss, tpr_loss)
    return fpr_loss, tpr_loss, auc_data