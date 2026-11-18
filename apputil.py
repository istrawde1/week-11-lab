import numpy as np
from sklearn.cluster import KMeans

def kmeans(X, k):
    model = KMeans(n_clusters=k, random_state=0, n_init=10)
    labels = model.fit_predict(X)
    centroids = model.cluster_centers_
    return centroids, labels
