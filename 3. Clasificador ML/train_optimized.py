"""
Autor: Ernesto Juarez Torres A01754887  
Fecha: 2025-05

Este script entrena dos variantes de SVM lineal (una con `class_weight=None` y otra con `"balanced"`),
realiza optimización de hiperparámetro `C` mediante GridSearchCV, y evalúa su desempeño usando AUC-ROC.
Los modelos y gráficas se guardan, junto con una tabla de métricas y un heatmap comparativo.

"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import roc_auc_score, RocCurveDisplay
from scipy.special import expit

from dataload import load_all_matrices

# ---------- Rutas ----------
BASE  = Path(__file__).parent
SPLIT = BASE / "splits"
OUT   = BASE / "out"
OUT.mkdir(exist_ok=True)

# ---------- Carga de datos ----------
X_full, y = load_all_matrices()
train_idx  = np.load(SPLIT / "train_idx.npy")
valid_idx  = np.load(SPLIT / "valid_idx.npy")
test_idx   = np.load(SPLIT / "test_idx.npy")
train_valid_idx = np.concatenate([train_idx, valid_idx])

# ---------- Anti-fuga ----------
def strip_leakage(X, y, idx):
    """
    Elimina características que aparecen exclusivamente en una clase.
    """
    Xtv  = X[idx]
    keep = ((Xtv[y[idx] == 1].sum(0).A1 > 0) &
            (Xtv[y[idx] == 0].sum(0).A1 > 0))
    return X[:, keep]

X = strip_leakage(X_full, y, train_valid_idx)

# ---------- Entrenamiento y evaluación ----------
def train_and_eval(name, class_weight):
    """
    Entrena un modelo SVM con GridSearchCV y evalúa AUC sobre test.
    Guarda el modelo y la curva ROC.
    """
    svm  = LinearSVC(dual=False, class_weight=class_weight,
                     max_iter=5000, random_state=42)
    grid = {"C": [0.01, 0.1, 1, 10]}
    cv   = StratifiedKFold(5, shuffle=True, random_state=42)
    gs   = GridSearchCV(svm, grid, scoring="roc_auc", cv=cv, n_jobs=-1)
    gs.fit(X[train_valid_idx], y[train_valid_idx])

    best   = gs.best_estimator_
    scores = expit(best.decision_function(X[test_idx]))
    auc    = roc_auc_score(y[test_idx], scores)

    print(f"{name:13s} AUC={auc:.4f}")

    RocCurveDisplay.from_predictions(y[test_idx], scores)
    plt.title(f"ROC – {name}")
    plt.savefig(OUT / f"roc_{name}.png", dpi=140)
    plt.close()

    joblib.dump(best, OUT / f"{name}.joblib")
    return {"modelo": name, "AUC": auc}

# Entrenar modelos
results = [
    train_and_eval("svm", None),
    train_and_eval("svm_balanced", "balanced"),
]

# ---------- Guardar métricas y heatmap ----------
df = pd.DataFrame(results).set_index("modelo")
df.to_csv(OUT / "metrics.csv", mode="a", header=not (OUT / "metrics.csv").exists())

sns.heatmap(df, annot=True, cmap="Blues", fmt
