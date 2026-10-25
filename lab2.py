# EX 1
import numpy as np
import matplotlib.pyplot as plt

print("Ex 1")

def leverage(X):
    H = X @ np.linalg.inv(X.T @ X) @ X.T
    return np.diag(H)

rng = np.random.default_rng(42)
def make_group(n, x_std, noise_mu, noise_std):
    x = rng.normal(0, x_std, n)
    y = 5*x + 3 + rng.normal(noise_mu, noise_std, n)
    return x, y

x_reg, y_reg = make_group(100, 1, 2, 2)
x_hx, y_hx = make_group(10, 10, 2, 2)
x_hy, y_hy = make_group(10, 2, 2, 10)
x_hb, y_hb = make_group(10, 10, 2, 10)

x = np.concatenate([x_reg, x_hx, x_hy, x_hb])
y = np.concatenate([y_reg, y_hx, y_hy, y_hb])
X = np.column_stack([x, np.ones(len(x))])
h = leverage(X)

plt.scatter(x_reg, y_reg, c='blue', label='reg')
plt.scatter(x_hx, y_hx, c='red', label='high x')
plt.scatter(x_hy, y_hy, c='green', label='high y')
plt.scatter(x_hb, y_hb, c='yellow', label='high both')
top = np.argsort(h)[-10:]
plt.scatter(x[top], y[top], s=150, facecolors='none', edgecolors='black')
plt.show()

def leverage_panel(ax, noise_mu, noise_std, title):
    x_reg, y_reg = make_group(100, 1, noise_mu, noise_std)
    x_hx, y_hx = make_group(10, 10, noise_mu, noise_std)
    x_hy, y_hy = make_group(10, 2, noise_mu, 10)
    x_hb, y_hb = make_group(10, 10, noise_mu, 10)

    x = np.concatenate([x_reg, x_hx, x_hy, x_hb])
    y = np.concatenate([y_reg, y_hx, y_hy, y_hb])
    X = np.column_stack([x, np.ones(len(x))])
    h = leverage(X)

    ax.scatter(x_reg, y_reg, c='blue', label='reg')
    ax.scatter(x_hx, y_hx, c='red', label='high x')
    ax.scatter(x_hy, y_hy, c='green', label='high y')
    ax.scatter(x_hb, y_hb, c='yellow', label='high both')
    top = np.argsort(h)[-10:]
    ax.scatter(x[top], y[top], s=150, facecolors='none', edgecolors='black')
    ax.set_title(title)

noise_levels = [0.5, 2, 5]
noise_means = [0, 5, 20]
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for i, noise_std in enumerate(noise_levels):
    leverage_panel(axes[0, i], 2, noise_std, f"mu = 2, sigma^2 = {noise_std**2}")
for i, noise_mu in enumerate(noise_means):
    leverage_panel(axes[1, i], noise_mu, 2, f"mu = {noise_mu}, sigma^2 = 4")
plt.show()


def make_group_2d(n, x_std, noise_mu, noise_std):
    x1 = rng.normal(0, x_std, n)
    x2 = rng.normal(0, x_std, n)
    y = 5*x1 + 2*x2 + 3 + rng.normal(noise_mu, noise_std, n)
    return x1, x2, y

# 2d case
fig, axes = plt.subplots(1, len(noise_levels), figsize=(15, 4))
for i, noise_std in enumerate(noise_levels):
    x1_reg, x2_reg, y_reg = make_group_2d(100, 1, 2, noise_std)
    x1_hx, x2_hx, y_hx = make_group_2d(10, 10, 2, noise_std)
    x1_hy, x2_hy, y_hy = make_group_2d(10, 2, 2, 10)
    x1_hb, x2_hb, y_hb = make_group_2d(10, 10, 2, 10)

    x1 = np.concatenate([x1_reg, x1_hx, x1_hy, x1_hb])
    x2 = np.concatenate([x2_reg, x2_hx, x2_hy, x2_hb])
    y = np.concatenate([y_reg, y_hx, y_hy, y_hb])
    X = np.column_stack([x1, x2, np.ones(len(x1))])
    h = leverage(X)
    print(f"2D, sigma^2 = {noise_std**2}: X shape {X.shape}, sum of leverages {h.sum():.3f}")

    axes[i].scatter(x1_reg, x2_reg, c='blue', label='reg')
    axes[i].scatter(x1_hx, x2_hx, c='red', label='high x')
    axes[i].scatter(x1_hy, x2_hy, c='green', label='high y')
    axes[i].scatter(x1_hb, x2_hb, c='orange', label='high both')
    top = np.argsort(h)[-10:]
    axes[i].scatter(x1[top], x2[top], s=150, facecolors='none', edgecolors='black', label='top 10 leverage')
    axes[i].set_xlabel('x1')
    axes[i].set_ylabel('x2')
    axes[i].set_title(f"2D, mu = 2, sigma^2 = {noise_std**2}")
axes[0].legend()
plt.show()

# EX2
from pyod.utils.data import generate_data_clusters
from pyod.models.knn import KNN
from sklearn.metrics import balanced_accuracy_score
import numpy as np
import matplotlib.pyplot as plt

print("Ex 2")

X_train, X_test, y_train, y_test = generate_data_clusters(n_train=400, n_test=200, n_clusters=2, n_features=2, contamination=0.1, random_state=42)


def plot_labels(ax, X, labels, title):
    normals = X[labels == 0]
    outliers = X[labels == 1]
    ax.scatter(normals[:, 0], normals[:, 1], c='blue')
    ax.scatter(outliers[:, 0], outliers[:, 1], c='red')
    ax.set_title(title)

for k in [1, 5, 200]:
    knn = KNN(n_neighbors=k, contamination=0.1)
    knn.fit(X_train)

    y_pred = knn.predict(X_test)
    ba_train = balanced_accuracy_score(y_train, knn.labels_)
    ba_test = balanced_accuracy_score(y_test, y_pred)
    print(f"ba train: {ba_train}")
    print(f"ba test: {ba_test}")

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    plot_labels(axes[0, 0], X_train, y_train, "train true")
    plot_labels(axes[0, 1], X_train, knn.labels_, "train pred")
    plot_labels(axes[1, 0], X_test, y_test, "test true")
    plot_labels(axes[1, 1], X_test, y_pred, "test pred")
    fig.suptitle(f"KNN, k = {k}: train BA {ba_train:.3f}, test BA {ba_test:.3f}")
    plt.show()

for k in [1, 3, 5, 10, 50, 180, 200, 300]:
    knn = KNN(n_neighbors=k, contamination=0.1)
    knn.fit(X_train)

    y_pred = knn.predict(X_test)
    ba_train = balanced_accuracy_score(y_train, knn.labels_)
    ba_test = balanced_accuracy_score(y_test, y_pred)

    print(f"k={k}: train BA {ba_train:.3f}, test BA {ba_test:.3f}")

# EX 3
from sklearn.datasets import make_blobs
from pyod.models.lof import LOF

print("Ex 3")

X, cluster = make_blobs(n_samples=[200, 100], centers=[(-10, -10), (10, 10)], cluster_std=[2, 6], n_features=2, random_state=42)

for k in [5, 20, 50]:
    knn = KNN(n_neighbors=k, contamination=0.07)
    lof = LOF(n_neighbors=k, contamination=0.07)
    knn.fit(X)
    lof.fit(X)
    print(f"k={k}")
    print("KNN flags in tight cluster:", knn.labels_[cluster == 0].sum())
    print("KNN flags in sparse cluster:", knn.labels_[cluster == 1].sum())
    print("LOF flags in tight cluster:", lof.labels_[cluster == 0].sum())
    print("LOF flags in sparse cluster:", lof.labels_[cluster == 1].sum())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    knn_pred = knn.labels_
    lof_pred = lof.labels_
    plot_labels(axes[0], X, knn_pred, "knn pred")
    plot_labels(axes[1], X, lof_pred, "lof pred")
    fig.suptitle(f"n_neighbors = {k}")
    plt.show()

# EX 4
from sklearn.model_selection import train_test_split
from pyod.utils.utility import standardizer
from pyod.models.combination import average, maximization

print("Ex 4")

data = np.load("6_cardio.npz")
X = data["X"]
y = data["y"]
print(f"X.shape: {X.shape}")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

X_train_n, X_test_n = standardizer(X_train, X_test)

contamination = np.sum(y == 1)/len(y)

train_scores = np.zeros((len(X_train), 10))
test_scores = np.zeros((len(X_test), 10))
for i, k in enumerate([30, 40, 50, 60, 70, 80, 90, 100, 110, 120]):
    knn = KNN(n_neighbors=k, contamination=contamination)
    knn.fit(X_train_n)
    train_predictions = knn.labels_
    test_predictions = knn.predict(X_test_n)
    print(f"k={k}: train BA {balanced_accuracy_score(y_train, train_predictions):.3f}, test BA {balanced_accuracy_score(y_test, test_predictions):.3f}")
    train_scores[:, i] = knn.decision_scores_
    test_scores[:, i] = knn.decision_function(X_test_n)

train_scores_n, test_scores_n = standardizer(train_scores, test_scores)

# AVG strategy
train_scores_avg = average(train_scores_n) # np.average(train_scores_n, axis=1)
test_scores_avg = average(test_scores_n) # np.average(test_scores_n, axis=1)
threshold = np.quantile(train_scores_avg, 1 - contamination)
train_anomalies = train_scores_avg > threshold
test_anomalies = test_scores_avg > threshold
print(f"AVG train BA {balanced_accuracy_score(y_train, train_anomalies):.3f}, test BA {balanced_accuracy_score(y_test, test_anomalies):.3f}")

# MAX strategy
train_scores_max = maximization(train_scores_n)
test_scores_max = maximization(test_scores_n)
threshold = np.quantile(train_scores_max, 1 - contamination)
train_anomalies = train_scores_max > threshold
test_anomalies = test_scores_max > threshold
print(f"MAX train BA {balanced_accuracy_score(y_train, train_anomalies):.3f}, test BA {balanced_accuracy_score(y_test, test_anomalies):.3f}")
