"""
Simple CART-style decision tree:
1. Recursively split data by feature/threshold pairs that maximize
   information gain (Gini impurity reduction).
2. Stop splitting when leaf is pure, depth limit reached, or samples
   fall below min_samples_split. Leaves hold majority class.
3. During prediction, traverse from root to leaf based on thresholds.
Designed to be short enough for interview coding exercises.
"""

import numpy as np
from collections import Counter
from typing import Optional, Tuple


class TreeNode:
    """Lightweight node for interview-style decision tree."""

    def __init__(self,
                 feature: Optional[int] = None,
                 threshold: Optional[float] = None,
                 left=None,
                 right=None,
                 *,
                 value: Optional[int] = None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # majority class at leaf

    def is_leaf(self) -> bool:
        return self.value is not None


class SimpleDecisionTree:
    """
    Small CART-style classifier supporting continuous features
    and binary splits. Depth-limited so you can explain every step.
    """

    def __init__(self, max_depth: int = 3, min_samples_split: int = 2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None

    def fit(self, x: np.ndarray, y: np.ndarray):
        self.root = self._build_tree(x, y, depth=0)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return np.array([self._traverse(row, self.root) for row in x])

    # ---------------- internal helpers ---------------- #
    def _build_tree(self, x: np.ndarray, y: np.ndarray, depth: int) -> TreeNode:
        # stop if pure, too small, or hit depth limit
        if (depth >= self.max_depth or
                len(set(y)) == 1 or
                x.shape[0] < self.min_samples_split):
            return TreeNode(value=self._majority_class(y))

        feature, threshold, gain = self._best_split(x, y)

        if gain == 0 or feature is None:
            return TreeNode(value=self._majority_class(y))

        left_mask = x[:, feature] <= threshold
        right_mask = ~left_mask

        left_child = self._build_tree(x[left_mask], y[left_mask], depth + 1)
        right_child = self._build_tree(x[right_mask], y[right_mask], depth + 1)

        return TreeNode(feature, threshold, left_child, right_child)

    def _best_split(self, x: np.ndarray, y: np.ndarray) -> Tuple[Optional[int], Optional[float], float]:
        # --- Interview hot spot: selecting feature/threshold with max information gain ---
        best_gain = 0.0
        best_feature = None
        best_threshold = None
        current_impurity = self._gini(y)

        n_samples, n_features = x.shape

        for feature in range(n_features):
            thresholds = np.unique(x[:, feature])
            for threshold in thresholds:
                left_mask = x[:, feature] <= threshold
                right_mask = ~left_mask
                if left_mask.sum() == 0 or right_mask.sum() == 0:
                    continue

                gain = self._information_gain(y, left_mask, right_mask, current_impurity)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    @staticmethod
    def _information_gain(y, left_mask, right_mask, current_impurity):
        n = len(y)
        left_impurity = SimpleDecisionTree._gini(y[left_mask])
        right_impurity = SimpleDecisionTree._gini(y[right_mask])
        weighted = (left_mask.sum() / n) * left_impurity + (right_mask.sum() / n) * right_impurity
        return current_impurity - weighted

    @staticmethod
    def _gini(labels: np.ndarray) -> float:
        # --- Interview hot spot: impurity calculation formula ---
        counts = Counter(labels)
        total = len(labels)
        return 1.0 - sum((count / total) ** 2 for count in counts.values())

    @staticmethod
    def _majority_class(labels: np.ndarray) -> int:
        # --- Interview hot spot: tie-breaking at leaf nodes ---
        counts = Counter(labels)
        return counts.most_common(1)[0][0]

    def _traverse(self, row: np.ndarray, node: TreeNode) -> int:
        # --- Interview hot spot: tree traversal during inference ---
        while not node.is_leaf():
            if row[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value


if __name__ == "__main__":
    # quick demo
    np.random.seed(0)
    X = np.random.randn(200, 2)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    model = SimpleDecisionTree(max_depth=3)
    model.fit(X, y)

    preds = model.predict(X)
    accuracy = (preds == y).mean()
    print("Training accuracy:", accuracy)
