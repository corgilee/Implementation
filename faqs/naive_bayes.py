import numpy as np
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score


class GaussianNaiveBayes:
    def __init__(self):
        self.class_priors = None
        self.class_means = None
        self.class_vars = None
        self.classes = None

    def fit(self, x: np.ndarray, y: np.ndarray):
        self.classes = np.unique(y)
        n_features = x.shape[1]

        self.class_priors = {}
        self.class_means = {}
        self.class_vars = {}

        for c in self.classes:
            x_c = x[y == c]
            self.class_priors[c] = x_c.shape[0] / x.shape[0]
            self.class_means[c] = x_c.mean(axis=0)
            # variance can underflow; add small eps for stability
            self.class_vars[c] = x_c.var(axis=0) + 1e-6

        # store as arrays for vectorized predict
        self.class_priors = np.log(
            np.array([self.class_priors[c] for c in self.classes])
        )
        self.class_means = np.vstack([self.class_means[c] for c in self.classes])
        self.class_vars = np.vstack([self.class_vars[c] for c in self.classes])

    # P(y | x) ∝ P(y) * Π_j P(x_j | y)
    # For Gaussian features:
    #   P(x_j | y=c) = (1 / sqrt(2πσ_cj^2)) * exp(-(x_j - μ_cj)^2 / (2σ_cj^2))
    # Taking logs yields log P(y|x) = log P(y) + Σ_j log P(x_j | y=c)

    def _log_likelihood(self, x: np.ndarray):
        x = x[None, :, :]  # [1, samples, features]
        mean = self.class_means[:, None, :]  # [classes, 1, features]
        var = self.class_vars[:, None, :]  # [classes, 1, features]

        coeff = -0.5 * np.sum(np.log(2 * np.pi * var), axis=2)  # [classes, 1]
        exponent = -0.5 * np.sum(((x - mean) ** 2) / var, axis=2)  # [classes, samples]
        return coeff + exponent

    def predict(self, x: np.ndarray):
        # expand dims so we broadcast over classes
        log_likelihood = self._log_likelihood(x)
        log_post = self.class_priors[:, None] + log_likelihood
        preds = np.argmax(log_post, axis=0)
        return self.classes[preds]


if __name__ == "__main__":
    x, y = make_classification(
        n_samples=1000,
        n_features=8,
        n_informative=6,
        n_redundant=2,
        random_state=42,
    )

    split = int(0.8 * len(x))
    x_train, x_test = x[:split], x[split:]
    y_train, y_test = y[:split], y[split:]

    model = GaussianNaiveBayes()
    model.fit(x_train, y_train)
    preds = model.predict(x_test)
    print("Naive Bayes accuracy:", accuracy_score(y_test, preds))
