# Clustering: K-Means

import numpy as np
from sklearn.cluster import KMeans

# Sample data
X = np.array([
    [1, 2], [1, 4], [1, 0],
    [10, 2], [10, 4], [10, 0]
])

# Model
kmeans = KMeans(n_clusters=2, random_state=0)
kmeans.fit(X)

# Output
print("Labels:", kmeans.labels_)
print("Centroids:", kmeans.cluster_centers_)