# tests/test_training.py
"""
Autor: Ernesto Juárez Torres A01754887
Fecha: 2025-05

Verifica que el pipeline de entrenamiento genera modelos con AUC > 0.5
sobre una muestra reducida del conjunto.
"""

import numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent / "3. Clasificador ML"))
from dataload import load_all_features_and_labels


def test_quick_auc_minimo():
    X, y = load_all_features_and_labels()
    n    = max(50, int(0.01 * X.shape[0]))  # 1% de la muestra
    rng  = np.random.RandomState(0)
    idx  = rng.choice(np.arange(X.shape[0]), n, replace=False)
    Xs, ys = X.iloc[idx], y[idx]

    clf = LogisticRegression(max_iter=500).fit(Xs, ys)
    auc = roc_auc_score(ys, clf.predict_proba(Xs)[:, 1])
    assert auc >= 0.5  # debe superar el azar
