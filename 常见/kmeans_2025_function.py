import numpy as np

# -----------------------
# Euclidean distance
# -----------------------
def euclidean_distance(a, b):
    # sqrt(sum((a - b)^2))
    return np.sqrt(np.sum((a - b) ** 2))


# -----------------------
# Assign each point to nearest centroid
# -----------------------
def assign_clusters(X, centroids):
    """
    X: (n_samples, n_features)
    centroids: (k, n_features)
    return: cluster index for each sample
    """
    labels = []

    for x in X:
        # compute distance to each centroid
        distances = [euclidean_distance(x, c) for c in centroids]
        # pick the closest centroid
        labels.append(np.argmin(distances))

    return np.array(labels)


# -----------------------
# Recompute centroids
# -----------------------
def update_centroids(X, labels, k):
    """
    X: data points
    labels: cluster assignment
    k: number of clusters
    """
    centroids = []

    for i in range(k):
        cluster_points = X[labels == i]

        # handle empty cluster
        if len(cluster_points) == 0:
            centroids.append(X[np.random.randint(0, len(X))])
        else:
            centroids.append(cluster_points.mean(axis=0))

    return np.array(centroids)


# -----------------------
# K-means main loop
# -----------------------
def kmeans(X, k, max_iters=100, tol=1e-4, random_state=78):
    np.random.seed(random_state)

    n_samples = X.shape[0]

    # 1) initialize centroids by sampling k points
    init_idx = np.random.choice(n_samples, k, replace=False)
    centroids = X[init_idx]

    for it in range(max_iters):
        # 2) assignment step
        labels = assign_clusters(X, centroids)

        # 3) update step
        new_centroids = update_centroids(X, labels, k)

        # 4) check convergence (centroid movement)
        shift = np.linalg.norm(new_centroids - centroids)

        centroids = new_centroids

        if shift < tol:
            # print(f"Converged at iteration {it+1}")
            break

    return centroids, labels


# -----------------------
# Example usage
# -----------------------
X = np.array([
    [0.2, 0.3, -0.5],
    [0.1, 0.4, -0.6],
    [2.0, 1.9,  2.1],
    [2.2, 2.1,  1.8],
    [-1.0, -0.8, -1.2],
    [-0.9, -1.1, -1.0],
])

centroids, labels = kmeans(X, k=3)

print("Centroids:")
print(centroids)
print("Labels:")
print(labels)
