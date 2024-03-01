'''

The K-means clustering algorithm is a partitioning method that aims to divide a dataset into clusters
, where each data point belongs to the cluster with the nearest centroid. 

k 是自己指定的
'''


import numpy as np






class KMeans:
    def __init__(self, n_clusters, max_iter=300, tol=1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol


    def fit(self, X):
        # Initialize centroids randomly
        self.centroids = X[np.random.choice(X.shape[0], self.n_clusters, replace=False)]
        
        for _ in range(self.max_iter):
            # Assign each data point to the nearest centroid
            labels = self._assign_labels(X)
            
            # Update centroids based on the mean of data points assigned to each cluster
            new_centroids = np.array([X[labels == k].mean(axis=0) for k in range(self.n_clusters)])
            
            # Check for convergence
            if np.allclose(self.centroids, new_centroids, atol=self.tol):
                break
            
            self.centroids = new_centroids
        
        self.labels_ = self._assign_labels(X)
        
        return self #这个写法要记住

    def _assign_labels(self, X):
        # Calculate distances between data points and centroids
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        print('distance', distances.shape)

        # Assign labels based on nearest centroid
        return np.argmin(distances, axis=0) # 在k个中心点中，找到最近的那个中心点的索引

np.random.seed(0)
x = np.random.randn(100, 2) # 100 points in 2D

# Instantiate and fit KMeans
kmeans = KMeans(n_clusters=3)
kmeans.fit(x)

# Get cluster centroids and labels
centroids = kmeans.centroids
labels = kmeans.labels_

print("Cluster centroids:")
print(centroids)
print("Cluster labels:")
print(labels)

