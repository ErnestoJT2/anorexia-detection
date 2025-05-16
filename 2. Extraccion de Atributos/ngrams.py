"""
Autor: Ernesto Juárez Torres
Fecha: 2025-05

Transforma un corpus textual en matriz de n-gramas (bi- y tri-gramas),
con padding si se generan menos columnas que `max_features`.
"""

from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix, hstack
from typing import List, Tuple

__all__ = ["ngram_features"]

def ngram_features(corpus: List[str], n: Tuple[int, int] = (2, 3), max_features: int = 8000) -> csr_matrix:
    """
    Genera matriz de n-gramas con padding.

    Parámetros:
    - corpus: Lista de documentos (tokens separados por espacio).
    - n: Rango de n-gramas a usar, ej. (2,3).
    - max_features: Límite máximo de columnas.

    Retorna:
    - Matriz dispersa CSR (n_docs, max_features).
    """
    vec = CountVectorizer(tokenizer=str.split, lowercase=False, max_features=max_features, ngram_range=n)
    X = vec.fit_transform(corpus)

    if X.shape[1] < max_features:
        extra = csr_matrix((X.shape[0], max_features - X.shape[1]), dtype=X.dtype)
        X = hstack([X, extra], format="csr")

    return X
