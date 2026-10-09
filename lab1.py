import pyod.utils.data as pyod
from matplotlib import pyplot

# ex 1
X_train, X_test, y_train, y_test = pyod.generate_data(
    n_train=400, n_test=100, n_features=2, contamination=0.1
)

inliers = X_train[y_train == 0]
outliers = X_train[y_train == 1]

pyplot.scatter(inliers[:, 0], inliers[:, 1], c='blue')
pyplot.scatter(outliers[:, 0], outliers[:, 1], c='red')

pyplot.show()

# ex 2
print("ex2")
from pyod.models.knn import KNN
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
knn = KNN(contamination=0.1)
knn.fit(X_train)

# for test data
predictions = knn.predict(X_test)
scores = knn.decision_function(X_test)

(tn, fp), (fn, tp) = confusion_matrix(y_test, predictions)

tpr = (tp)/(tp + fn)
tnr = (tn)/(tn + fp)
balanced_acc = (tpr + tnr) / 2
print(f"balanced acc was: {balanced_acc}")

fpr, tpr, _ = roc_curve(y_test, scores)
pyplot.plot(fpr, tpr)
pyplot.show()

# for train data
predictions = knn.predict(X_train)
scores = knn.decision_function(X_train)

(tn, fp), (fn, tp) = confusion_matrix(y_train, predictions)

tpr = (tp)/(tp + fn)
tnr = (tn)/(tn + fp)
balanced_acc = (tpr + tnr) / 2
print(f"balanced acc was: {balanced_acc}")

fpr, tpr, _ = roc_curve(y_train, scores)
pyplot.plot(fpr, tpr)
pyplot.show()

#ex 3
print("ex3")
import numpy as np
X_train, X_test, y_train, y_test = pyod.generate_data(
    n_train=1000, n_test=0, n_features=1, contamination=0.1
)

z = np.abs((X_train - np.mean(X_train)) / np.std(X_train))
threshold = np.quantile(z, 1 - 0.1)
y_pred = (z > threshold)

(tn, fp), (fn, tp) = confusion_matrix(y_train, y_pred)
ba = (tp / (tp + fn) + tn / (tn + fp)) / 2
print(f"threshold: {threshold}, balanced acc: {ba}")

# ex4
print("ex4")
from scipy.linalg import solve_triangular
mean = [2, -1]
sigma = [[3, 1.5], [1.5, 2]]

L = np.linalg.cholesky(sigma)

# inliers
x = np.random.standard_normal((900, 2))
y_inliers = x @ L.T + mean

# outliers
y_outliers = np.random.uniform(low=-15, high=15, size=(100, 2))

X_train = np.vstack([y_inliers, y_outliers])
y_train = np.hstack((np.zeros(900), np.ones(100)))

z_per_feature = np.abs((X_train - np.mean(X_train, axis=0)) / np.std(X_train, axis=0))
z_scores = np.mean(z_per_feature, axis=1) 

threshold = np.quantile(z_scores, 1 - 0.1)
y_pred = (z_scores > threshold).astype(int)

(tn, fp), (fn, tp) = confusion_matrix(y_train, y_pred)
ba = (tp / (tp + fn) + tn / (tn + fp)) / 2
print(f"threshold: {threshold:.4f}, balanced acc: {ba:.4f}")



