import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

rng = np.random.default_rng(42)

X_train, _ = make_blobs(n_samples=500, n_features=2, centers=[[0, 0]],
                        cluster_std=1.0, random_state=42)

n_proj = 5
W = rng.multivariate_normal((0, 0), [[1, 0], [0, 1]], size=n_proj)
W = W / np.linalg.norm(W, axis=1, keepdims=True)

n_bins = 20
hist_range = (-5, 5)

probs = []
edges = []
for i in range(n_proj):
    z = X_train @ W[i]
    counts, e = np.histogram(z, bins=n_bins, range=hist_range)
    probs.append(counts / 500)
    edges.append(e)


def loda_score(X):
    per_proj = np.zeros((len(X), n_proj))
    for i in range(n_proj):
        z = X @ W[i]
        bin_idx = np.clip(np.digitize(z, edges[i]) - 1, 0, n_bins - 1)
        per_proj[:, i] = probs[i][bin_idx]
    return per_proj.mean(axis=1)


X_test = rng.uniform([-3, -3], [3, 3], size=(500, 2))
scores = loda_score(X_test)

plt.scatter(X_test[:, 0], X_test[:, 1], c=scores, cmap="viridis")
plt.colorbar(label="mean bin probability (low = anomalous)")
plt.title(f"loda {n_bins} bins")
plt.show()

