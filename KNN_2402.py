import numpy as np
from collections import Counter

def euclidean_distance(point1, point2):
    return np.sqrt(np.sum((point1 - point2) ** 2))

class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        # Compute distances between x and all examples in the training set
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # Sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.k]
        # Extract the labels of the k nearest neighbor training samples
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Return the most common class label
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]


# Test case
if __name__ == "__main__":
    # Create a simple dataset
    X_train = np.array([
        [1, 2],
        [2, 3],
        [3, 3],
        [6, 5],
        [7, 7],
        [8, 7]
    ])
    y_train = np.array([0, 0, 0, 1, 1, 1])  # 0 for Class A, 1 for Class B

    # Create and train the model
    classifier = KNN(k=3)
    classifier.fit(X_train, y_train)

    # Make a prediction for a new data point
    X_test = np.array([[5, 5]])
    prediction = classifier.predict(X_test)

    print(f"The predicted class for the point {X_test[0]} is: {prediction[0]}")
    # The output will depend on the value of k and the distribution of the data points