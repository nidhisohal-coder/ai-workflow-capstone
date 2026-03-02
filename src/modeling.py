import numpy as np

class LastValueRegressor:
    """Predicts a constant equal to the last observed y seen during fit()."""
    def fit(self, X, y):
        y = np.asarray(y)
        self.last_ = float(y[-1])
        return self

    def predict(self, X):
        return np.full(len(X), self.last_, dtype=float)