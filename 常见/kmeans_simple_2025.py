import numpy as np


class SimpleKMeans:
    """Interview-friendly K-Means with minimal moving parts."""

    def __init__(self, n_clusters: int, max_iter: int = 100, tol: float = 1e-4):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.centroids = None

    def fit(self, x: np.ndarray):
        np.random.seed(0)
        indices = np.random.choice(x.shape[0], size=self.n_clusters, replace=False)
        self.centroids = x[indices]

        for _ in range(self.max_iter):
            labels = self.predict(x)
            centroid_list = []
            for k in range(self.n_clusters):
                cluster_points = x[labels == k]
                centroid_list.append(cluster_points.mean(axis=0))
            
            new_centroids = np.vstack(centroid_list) # (n_cluster, n_features)

            if np.linalg.norm(self.centroids - new_centroids) < self.tol:
                break
            self.centroids = new_centroids

    def predict(self, x: np.ndarray) -> np.ndarray:
        # x: [samples, features] -> [samples, 1, features]
        samples = x[:, None, :]
        # centroids: [clusters, features] -> [1, clusters, features]
        centroids = self.centroids[None, :, :]
        # squared distances via broadcasting: result [samples, clusters]
        dists = ((samples - centroids) ** 2).sum(axis=2) #[n_samples, n_clusters]
        return np.argmin(dists, axis=1)


if __name__ == "__main__":
    data = np.random.randn(200, 2)
    model = SimpleKMeans(n_clusters=3)
    model.fit(data)
    print("Centroids:\n", model.centroids)
    print("Cluster counts:", np.bincount(model.predict(data)))
