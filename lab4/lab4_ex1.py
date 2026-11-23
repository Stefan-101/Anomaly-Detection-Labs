import matplotlib.pyplot as plt
from pyod.utils.data import generate_data
from pyod.models.ocsvm import OCSVM
from pyod.models.deep_svdd import DeepSVDD
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

CONT = 0.15

X_train, X_test, y_train, y_test = generate_data(n_train=300, n_test=200, n_features=3, contamination=CONT)


def evaluate(model, name):
    model.fit(X_train)
    y_train_pred = model.labels_
    y_test_pred = model.predict(X_test)
    scores = model.decision_function(X_test)
    print(f"{name:12s} BA={balanced_accuracy_score(y_test, y_test_pred):.3f} "
          f"AUC={roc_auc_score(y_test, scores):.3f}")
    plot_4(y_train_pred, y_test_pred, name)


def scatter3d(ax, X, y, title):
    ax.scatter(X[y == 0, 0], X[y == 0, 1], X[y == 0, 2], c="tab:blue", s=8, label="inlier")
    ax.scatter(X[y == 1, 0], X[y == 1, 1], X[y == 1, 2], c="tab:red", s=8, label="outlier")
    ax.set_title(title)


def plot_4(y_train_pred, y_test_pred, name):
    fig = plt.figure(figsize=(10, 9))
    panels = [
        (X_train, y_train, "train - ground truth"),
        (X_train, y_train_pred, "train - predicted"),
        (X_test, y_test, "test - ground truth"),
        (X_test, y_test_pred, "test - predicted"),
    ]
    for i, (X, y, title) in enumerate(panels):
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")
        scatter3d(ax, X, y, title)
    fig.suptitle(name)
    plt.tight_layout()


evaluate(OCSVM(kernel='linear', contamination=CONT), "OCSVM linear")

evaluate(OCSVM(kernel='rbf', contamination=CONT), "OCSVM rbf")

evaluate(DeepSVDD(n_features=3, contamination=CONT, epochs=50), "DeepSVDD")

plt.show()
