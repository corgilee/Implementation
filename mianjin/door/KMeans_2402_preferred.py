import numpy as np

class KMeans:
    def __init__(self,n_clusters, max_iter=300, tol=1e-4):
        self.n_clusters=n_clusters
        self.max_iter=max_iter
        self.tol=tol

    def fit(self, x):
        # 先 initiate 中心
        self.centroids= x[np.random.choice(x.shape[0],size=self.n_clusters)]
        # O(n_cluster)

        for _ in range(self.max_iter):
            # O(max_iter)
            # assign label
            self.label=self._assign_labels(x)

            # 根据label 找到新的中心
            new_centroids=[x[self.label==k].mean(axis=0) for k in range(self.n_clusters)]

            # O(n_cluster * d)
            if np.allclose(self.centroids,new_centroids,rtol=self.tol):
                break

            self.centroids=new_centroids

        self.label=self._assign_labels(x)

    def _assign_labels(self,x):
        # for each cluster, the complexity is n*d (number of data * dimension)
        # therefore, total complexity is n_centroid * n * d
        dists=[]
        for i in range(self.n_clusters):
            dist=np.sqrt(np.sum((x-self.centroids[i])**2,axis=1)) #这里一定要有axis=1，因为x是multi_row, centroids 是一个点
            dists.append(dist)

        res=np.argmin(dists,axis=0) # 这里要有axis=0， dists has n_clusters rows
        return res


# Time complexity: Max_iteration * n_cluster * n (number of datapoints) * d 
np.random.seed(0)
x = np.random.randn(100, 2) # 100 points in 2D

# Instantiate and fit KMeans

kmeans = KMeans(n_clusters=2)
kmeans.fit(x)
print(kmeans.label)


kmeans = KMeans(n_clusters=3)
kmeans.fit(x)
print(kmeans.label)


# Get cluster centroids and labels



