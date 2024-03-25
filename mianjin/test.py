import numpy as np

from sklearn.datasets import make_classification
import numpy as np

# Generate a more complex dataset
X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, n_informative=15, n_redundant=5, random_state=42)


# Splitting the dataset into training and testing sets
split_ratio = 0.8  # 80% for training, 20% for testing
split_index = int(X.shape[0] * split_ratio)

X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

print(y_test)


