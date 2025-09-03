import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression, Perceptron, RidgeClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier,
    BaggingClassifier,
)
from sklearn.svm import SVC, LinearSVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB, BernoulliNB
from sklearn.neural_network import MLPClassifier

from sklearn.datasets import load_iris

data = load_iris(as_frame=True)
df = data.frame
df["target"] = data.target

X = df.drop("target", axis=1)
y = df["target"]

if y.dtype == "object":
    y = LabelEncoder().fit_transform(y)

# ordinal encoder
# one hot encoder
X = pd.get_dummies(X, drop_first=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# min max scaler or standard scaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

modelos = {
    # "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(ccp_alpha=[alphas for alphas in np.arange(0, 0.1, 0.01)]), #found best ccp_alpha with GridSearchCV
    # "Extra Tree": ExtraTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    # "Gradient Boosting": GradientBoostingClassifier(),
    # "AdaBoost": AdaBoostClassifier(),
    # "Bagging": BaggingClassifier(),
    "SVM": SVC(kernel="rbf", max_iter=2000),
    # "Linear SVM": LinearSVC(max_iter=2000),
    "KNN": KNeighborsClassifier(n_neighbors=5), #Test with different n_neighbors
    # "GaussianNB": GaussianNB(),
    # "BernoulliNB": BernoulliNB(),
    "MLP Neural Net": MLPClassifier(max_iter=2000), 
    # "Perceptron": Perceptron(max_iter=1000),
    # "Ridge Classifier": RidgeClassifier(),
}

resultados = []

#GridSearchCV for hyperparameter tuning
for nome, modelo in modelos.items():
    try:
        modelo.fit(X_train, y_train)
        y_pred = modelo.predict(X_test)

        #classification report
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted")
        recall = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        resultados.append(
            {
                "Modelo": nome,
                "Accuracy": accuracy,
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1,
            }
        )
    except Exception as e:
        resultados.append(
            {
                "Modelo": nome,
                "Accuracy": None,
                "Precision": None,
                "Recall": None,
                "F1-Score": None,
                "Erro": str(e),
            }
        )

resultados_df = pd.DataFrame(resultados)

#Classification report for the best model
melhores = resultados_df.sort_values(by="F1-Score", ascending=False)

print("Melhores modelos ordenados: ")
print(melhores)
