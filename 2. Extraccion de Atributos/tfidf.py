"""
Autor: Ernesto Juárez Torres
Fecha: 2025-05

Convierte un corpus en una matriz TF-IDF de unigramas, con padding si es necesario.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix, hstack
from typing import List

__all__ = ["tfidf_features"]

def tfidf_features(corpus: List[str], max_features: int = 5000) -> csr_matrix:
    """
    Genera matriz TF-IDF (unigramas) con padding a tamaño fijo.

    Parámetros:
    - corpus: Lista de textos tokenizados por espacio.
    - max_features: Número máximo de columnas (vocabulario).

    Retorna:
    - Matriz dispersa CSR (n_docs, max_features).
    """
    vec = TfidfVectorizer(tokenizer=str.split, lowercase=False, max_features=max_features)
    X = vec.fit_transform(corpus)

    if X.shape[1] < max_features:
        extra = csr_matrix((X.shape[0], max_features - X.shape[1]), dtype=X.dtype)
        X = hstack([X, extra], format="csr")

    return X
