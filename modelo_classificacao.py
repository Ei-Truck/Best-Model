import pandas as pd
import numpy as np

import warnings
from sklearn.exceptions import ConvergenceWarning

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    LabelEncoder,
    OrdinalEncoder,
    OneHotEncoder,
)
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

warnings.filterwarnings("ignore", category=ConvergenceWarning)

from sklearn.datasets import load_iris

data = load_iris(as_frame=True)
df = data.frame
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

if y.dtype == "object":
    y = LabelEncoder().fit_transform(y)

encoders = {
    "OrdinalEncoder": OrdinalEncoder(),
    "OneHotEncoder": OneHotEncoder(drop="first", sparse_output=False),
    "GetDummies": None,
}

scalers = {
    "StandardScaler": StandardScaler(),
    "MinMaxScaler": MinMaxScaler()
}

modelos = {
    "Decision Tree": (
        DecisionTreeClassifier(class_weight="balanced"),
        {
            "criterion": ["gini", "entropy", "log_loss"],
            "splitter": ["best", "random"],
            "max_depth": [None, 3, 5, 10, 20, 30],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 4, 6],
            "max_features": [None, "sqrt", "log2"],
            "ccp_alpha": [0.0, 0.01, 0.05, 0.1],
        },
    ),
    "Random Forest": (
        RandomForestClassifier(class_weight="balanced"),
        {
            "n_estimators": [50, 100, 200, 300, 500],
            "criterion": ["gini", "entropy", "log_loss"],
            "max_depth": [None, 5, 10, 20, 30],
            "min_samples_split": [2, 5, 10, 20],
            "min_samples_leaf": [1, 2, 4, 6],
            "max_features": ["sqrt", "log2", None],
            "bootstrap": [True, False],
        },
    ),
    "SVM": (
        SVC(),
        {
            "C": [0.01, 0.1, 1, 10, 100],
            "kernel": ["linear", "rbf", "poly", "sigmoid"],
            "gamma": ["scale", "auto"],
            "degree": [2, 3, 4, 5],
            "shrinking": [True, False],
        },
    ),
    "KNN": (
        KNeighborsClassifier(),
        {
            "n_neighbors": [3, 5, 7, 9, 11, 15],
            "weights": ["uniform", "distance"],
            "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
            "p": [1, 2],
        },
    ),
    "MLP Neural Net": (
        MLPClassifier(max_iter=2000, early_stopping=True),
        {
            "hidden_layer_sizes": [
                (50,), (100,), (50, 50), (100, 50),
                (50, 100, 50), (100, 100), (200,), (100, 100, 50),
            ],
            "activation": ["tanh", "relu", "logistic"],
            "solver": ["adam", "sgd", "lbfgs"],
            "alpha": [0.0001, 0.001, 0.01, 0.1],
            "learning_rate": ["constant", "adaptive"],
            "learning_rate_init": [0.0001, 0.001, 0.01],
        },
    ),
}

resultados = []

for enc_name, encoder in encoders.items():
    if enc_name == "GetDummies":
        X_enc = pd.get_dummies(X, drop_first=True)
    else:
        X_enc = encoder.fit_transform(X)
        if isinstance(X_enc, np.ndarray):
            X_enc = pd.DataFrame(X_enc)

    for sc_name, scaler in scalers.items():
        X_train, X_test, y_train, y_test = train_test_split(
            X_enc, y, test_size=0.3, random_state=42
        )

        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        for model_name, (modelo, param_grid) in modelos.items():
            try:
                grid = GridSearchCV(
                    modelo, param_grid, cv=5, scoring="f1_weighted", n_jobs=-1
                )
                grid.fit(X_train, y_train)
                y_pred = grid.predict(X_test)

                resultados.append({
                    "Encoder": enc_name,
                    "Scaler": sc_name,
                    "Modelo": model_name,
                    "Melhores Params": grid.best_params_,
                    "Accuracy": accuracy_score(y_test, y_pred),
                    "Precision": precision_score(y_test, y_pred, average="weighted"),
                    "Recall": recall_score(y_test, y_pred, average="weighted"),
                    "F1-Score": f1_score(y_test, y_pred, average="weighted"),
                })

            except Exception as e:
                resultados.append({
                    "Encoder": enc_name,
                    "Scaler": sc_name,
                    "Modelo": model_name,
                    "Melhores Params": None,
                    "Accuracy": None,
                    "Precision": None,
                    "Recall": None,
                    "F1-Score": None,
                    "Erro": str(e)
                })

resultados_df = pd.DataFrame(resultados)
melhores = resultados_df.sort_values(by="F1-Score", ascending=False)

print("Ranking final:")
print(melhores)
