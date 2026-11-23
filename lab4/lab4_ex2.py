import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.metrics import balanced_accuracy_score

data = np.load("../6_cardio.npz")
X, y = data["X"], data["y"]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.4, stratify=y, random_state=0)

y_train_sk = -2 * y_train + 1
y_test_sk = -2 * y_test + 1

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("ocsvm", OneClassSVM()),
])

cont = y_train.mean()
param_grid = [
    {"ocsvm__kernel": ["linear"],
     "ocsvm__nu": [0.05, cont, 0.2, 0.5]},
    {"ocsvm__kernel": ["rbf", "sigmoid"],
     "ocsvm__gamma": ["scale", 0.01, 0.1, 1],
     "ocsvm__nu": [0.05, cont, 0.2, 0.5]},
]

grid = GridSearchCV(pipe, param_grid, scoring="balanced_accuracy", cv=5)
grid.fit(X_train, y_train_sk)

print("best params:", grid.best_params_)
print("CV BA:", round(grid.best_score_, 3))
y_test_pred = grid.predict(X_test)
print("test BA:", round(balanced_accuracy_score(y_test_sk, y_test_pred), 3))
