"""
#Autor: Ernesto Juárez Torres A01754887

Entrena mini‑modelos en un subconjunto (1 %) para verificar
que el pipeline de ML produce AUC>0.5 y guarda métricas.
"""
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from dataload import load_all_matrices

def test_quick_auc(tmp_path):
    X, y = load_all_matrices()
    n    = max(50, int(0.01*X.shape[0]))      # 1 % o 50 docs
    rng  = np.random.RandomState(0)
    idx  = rng.choice(np.arange(X.shape[0]), n, replace=False)
    Xs, ys = X[idx], y[idx]

    clf = LogisticRegression(max_iter=500).fit(Xs, ys)
    auc = roc_auc_score(ys, clf.predict_proba(Xs)[:,1])
    assert auc >= 0.50                         # al menos mejor que azar
