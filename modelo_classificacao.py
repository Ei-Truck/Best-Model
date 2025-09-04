import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder, OrdinalEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.datasets import load_iris

data = load_iris(as_frame=True)
df = data.frame
df["target"] = data.target

X = df.drop("target", axis=1)
y = df["target"]

if y.dtype == "object":
    y = LabelEncoder().fit_transform(y)

encoders = {
    "OrdinalEncoder": OrdinalEncoder(),
    "OneHotEncoder": OneHotEncoder(drop="first", sparse_output=False),
    "GetDummies": None
}

scalers = {
    "StandardScaler": StandardScaler(),
    "MinMaxScaler": MinMaxScaler()
}

modelos = {
    "Decision Tree": (DecisionTreeClassifier(), {
        "max_depth": [None, 3, 5, 10],
        "ccp_alpha": [0.0, 0.01, 0.05],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    }),
    "Random Forest": (RandomForestClassifier(), {
        "n_estimators": [50, 100, 200, 300],
        "max_depth": [None, 5, 10, 20],
        "max_features": ["sqrt", "log2", None],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4]
    }),
    "SVM": (SVC(), {
        "C": [0.01, 0.1, 1, 10, 100],
        "kernel": ["linear", "rbf", "poly", "sigmoid"],
        "gamma": ["scale", "auto"]
    }),
    "KNN": (KNeighborsClassifier(), {
        "n_neighbors": [3, 5, 7, 9, 11],
        "weights": ["uniform", "distance"],
        "p": [1, 2]
    }),
    "MLP Neural Net": (MLPClassifier(max_iter=2000), {
        "hidden_layer_sizes": [(50,), (100,), (50, 50), (100, 50), (50, 100, 50)],
        "alpha": [0.0001, 0.001, 0.01],
        "learning_rate_init": [0.001, 0.01]
    })
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
                grid = GridSearchCV(modelo, param_grid, cv=5, scoring="f1_weighted", n_jobs=-1)
                grid.fit(X_train, y_train)

                y_pred = grid.predict(X_test)

                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred, average="weighted")
                recall = recall_score(y_test, y_pred, average="weighted")
                f1 = f1_score(y_test, y_pred, average="weighted")

                resultados.append({
                    "Encoder": enc_name,
                    "Scaler": sc_name,
                    "Modelo": model_name,
                    "Melhores Params": grid.best_params_,
                    "Accuracy": accuracy,
                    "Precision": precision,
                    "Recall": recall,
                    "F1-Score": f1
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
