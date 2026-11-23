import time
import scipy.io as sio
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from pyod.utils.utility import standardizer
from pyod.models.ocsvm import OCSVM
from pyod.models.deep_svdd import DeepSVDD

data = sio.loadmat("../shuttle.mat")
X = data["X"].astype(float)
y = data["y"].ravel()
cont = y.mean()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, stratify=y)
X_train, X_test = standardizer(X_train, X_test)


def report(model, name):
    t = time.time()
    model.fit(X_train)
    y_pred = model.predict(X_test)
    ba = balanced_accuracy_score(y_test, y_pred)
    scores = model.decision_function(X_test)
    auc = roc_auc_score(y_test, scores)
    print(f"{name:28s} BA={ba:.3f} AUC={auc:.3f} ({time.time() - t:.0f}s)")


report(OCSVM(contamination=cont), "OCSVM rbf")

architectures = [
    [64, 32],
    [32, 16, 8],
    [128, 64, 32]
]
for h in architectures:
    report(DeepSVDD(n_features=X_train.shape[1], hidden_neurons=h, contamination=cont, epochs=20, verbose=0), f"DeepSVDD {h}")
