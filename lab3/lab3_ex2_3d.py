import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from pyod.models.iforest import IForest
from pyod.models.dif import DIF
from pyod.models.loda import LODA

rng = np.random.default_rng(42)

X_train, _ = make_blobs(n_samples=[500, 500], centers=[(0, 10, 0), (10, 0, 10)], cluster_std=1.0, random_state=42)

X_test = rng.uniform(-10, 20, size=(1000, 3))

models = {
    "IForest": IForest(contamination=0.02, random_state=42),
    "DIF": DIF(contamination=0.02, random_state=42),
    "LODA": LODA(contamination=0.02, random_state=42),
}

fig = plt.figure(figsize=(18, 6))
for k, (name, model) in enumerate(models.items(), start=1):
    ax = fig.add_subplot(1, len(models), k, projection="3d")

    model.fit(X_train)
    scores = model.decision_function(X_test)

    sc = ax.scatter(X_test[:, 0], X_test[:, 1], X_test[:, 2], c=scores, cmap="viridis", s=10)
    ax.scatter(X_train[:, 0], X_train[:, 1], X_train[:, 2], c="red", s=2, alpha=0.2)
    ax.set_title(name)
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    fig.colorbar(sc, ax=ax, shrink=0.6, label="anomaly score (high = anomalous)")

plt.tight_layout()
plt.show()
