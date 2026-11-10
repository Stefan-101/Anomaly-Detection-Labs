import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from pyod.models.iforest import IForest
from pyod.models.dif import DIF
from pyod.models.loda import LODA

rng = np.random.default_rng(42)

X_train, _ = make_blobs(n_samples=[500, 500], centers=[(10, 0), (0, 10)], cluster_std=1.0, random_state=42)

X_test = rng.uniform([-10, -10], [20, 20], (1000, 2))

models = {
    "IForest": IForest(contamination=0.02, random_state=42),
    "DIF [16, 8]": DIF(contamination=0.02, hidden_neurons=[16, 8], random_state=42),
    "DIF [64, 32]": DIF(contamination=0.02, hidden_neurons=[64, 32], random_state=42),
    "DIF [500, 100]": DIF(contamination=0.02, random_state=42),
    "LODA 5 bins": LODA(contamination=0.02, n_bins=5, random_state=42),
    "LODA 10 bins": LODA(contamination=0.02, n_bins=10, random_state=42),
    "LODA 30 bins": LODA(contamination=0.02, n_bins=30, random_state=42),
}

n_cols = 3
n_rows = -(-len(models) // n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows), squeeze=False)
for ax in axes.flat[len(models):]:
    ax.axis("off")
for ax, (name, model) in zip(axes.flat, models.items()):
    model.fit(X_train)
    scores = model.decision_function(X_test)

    sc = ax.scatter(X_test[:, 0], X_test[:, 1], c=scores, cmap="viridis", s=12)
    ax.scatter(X_train[:, 0], X_train[:, 1], c="red", s=2, alpha=0.3)
    ax.set_title(name)
    ax.set_aspect("equal")
    fig.colorbar(sc, ax=ax, label="anomaly score (high = anomalous)")

plt.tight_layout()
plt.show()
