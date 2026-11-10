import numpy as np
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from pyod.utils.utility import standardizer
from pyod.models.iforest import IForest
from pyod.models.loda import LODA
from pyod.models.dif import DIF

data = sio.loadmat("shuttle.mat")
X = data["X"].astype(float)
y = data["y"].ravel()
contamination = y.mean()
print(X.shape, "anomaly fraction:", round(contamination, 4))


def make_models(seed):
    return {
        "IForest": IForest(contamination=contamination, random_state=seed),
        "LODA": LODA(contamination=contamination, random_state=seed),
        "DIF": DIF(contamination=contamination, random_state=seed),
    }


n_splits = 10
results = {name: {"BA": [], "AUC": []} for name in make_models(0)}

for split in range(n_splits):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, stratify=y)

    X_train, X_test = standardizer(X_train, X_test)

    for name, model in make_models(split).items():
        model.fit(X_train)
        y_pred = model.predict(X_test)
        scores = model.decision_function(X_test)

        results[name]["BA"].append(balanced_accuracy_score(y_test, y_pred))
        results[name]["AUC"].append(roc_auc_score(y_test, scores))
    print(f"split {split} done")

print()
for name, r in results.items():
    print(f"{name:8s}  BA {np.mean(r['BA']):.3f}"
          f"   AUC {np.mean(r['AUC']):.3f}")
