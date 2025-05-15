#Autor: Ernesto Juárez Torres A01754887

from dataload import load_all_matrices

def test_matrices_dim():
    X, y = load_all_matrices()
    assert X.shape[0] == len(y) > 0
    # TF‑IDF + BoW + N‑grams + 6 rasgos densos ⇒ > 6 columnas
    assert X.shape[1] > 6
