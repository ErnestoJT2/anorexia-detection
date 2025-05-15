"""
Autor: Ernesto Juarez Torres A01754887  
Fecha: 2025-05

Este script entrena un clasificador Random Forest como línea base para detección de riesgo de anorexia.
Aplica un filtro de anti-fuga local para eliminar atributos que aparecen exclusivamente en una sola clase,
evitando así sobreajuste artificial. Evalúa con AUC y F1 sobre el conjunto de prueba y guarda los resultados.

"""

from pathlib import Path
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, RocCurveDisplay

# ---------- Paths ----------
BASE  = Path(__file__).parent
SPLIT = BASE / "splits"
OUT   = BASE / "out"
OUT.mkdir(exist_ok=True)

# ---------- Carga de datos ----------
from dataload import load_all_matrices
X_full, y = load_all_matrices()

train_idx = np.load(SPLIT / "train_idx.npy")
valid_idx = np.load(SPLIT / "valid_idx.npy")
test_idx  = np.load(SPLIT / "test_idx.npy")
train_valid_idx = np.concatenate([train_idx, valid_idx])

# ---------- Anti-fuga de información ----------
def strip_leakage(X, y, idx):
    """
    Elimina atributos que aparecen solo en una clase del conjunto train+valid.
    """
    Xtv = X[idx]
    keep = ((Xtv[y[idx] == 1].sum(0).A1 > 0) &
            (Xtv[y[idx] == 0].sum(0).A1 > 0))
    return X[:, keep]

X = strip_leakage(X_full, y, train_valid_idx)

# ---------- Entrenamiento ----------
rf = RandomForestClassifier(
    n_estimators=300,
    max_features="sqrt",
    n_jobs=-1,
    random_state=42
).fit(X[train_valid_idx], y[train_valid_idx])

# ---------- Evaluación ----------
proba = rf.predict_proba(X[test_idx])[:, 1]
pred  = rf.predict(X[test_idx])
auc   = roc_auc_score(y[test_idx], proba)
f1    = f1_score(y[test_idx], pred, average="macro")

print(f"AUC baseline (RF): {auc:.4f} | F1: {f1:.4f}")

# ---------- Salida ----------
RocCurveDisplay.from_predictions(y[test_idx], proba)
plt.title("ROC – RandomForest baseline")
plt.savefig(OUT / "roc_random_forest.png", dpi=140)
plt.close()

pd.DataFrame([{"modelo": "random_forest", "AUC": auc, "F1": f1}]) \
  .to_csv(OUT / "metrics.csv", index=False)

joblib.dump(rf, OUT / "rf_baseline.joblib")
