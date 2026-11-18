from time import time

import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans


def kmeans(X, k):
    """Cluster the rows of X into k groups.

    Return the cluster centroids and the assigned label for each row.
    """
    model = KMeans(n_clusters=k, random_state=0, n_init=10)
    labels = model.fit_predict(X)
    centroids = model.cluster_centers_
    return centroids, labels


diamonds = sns.load_dataset("diamonds")
diamonds_numeric = diamonds.select_dtypes(include="number")


def kmeans_diamonds(n, k):
    """Cluster the first n numerical rows of the diamonds dataset."""
    X = diamonds_numeric.iloc[:n].to_numpy()
    return kmeans(X, k)


def kmeans_timer(n, k, n_iter=5):
    """Return the average runtime of n_iter k-means runs, in seconds."""
    runtimes = []

    for _ in range(n_iter):
        start = time()
        kmeans_diamonds(n, k)
        elapsed = time() - start
        runtimes.append(elapsed)

    return sum(runtimes) / n_iter
