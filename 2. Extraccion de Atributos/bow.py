"""
Autor: Ernesto Juárez Torres
Fecha: 2025-05

Transforma un corpus textual en una matriz dispersa Bag of Words (unigramas),
rellenando con ceros si el número de características encontradas es menor a `max_features`.
"""

from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix, hstack
from typing import List

__all__ = ["bow_features"]

def bow_features(corpus: List[str], max_features: int = 5000) -> csr_matrix:
    """
    Genera matriz BoW (unigramas) con padding a tamaño fijo.

    Parámetros:
    - corpus: Lista de textos preprocesados (tokens separados por espacio).
    - max_features: Número máximo de características (columnas).

    Retorna:
    - Matriz dispersa CSR de tamaño (n_docs, max_features).
    """
    vec = CountVectorizer(tokenizer=str.split, lowercase=False, max_features=max_features)
    X = vec.fit_transform(corpus)

    if X.shape[1] < max_features:
        extra = csr_matrix((X.shape[0], max_features - X.shape[1]), dtype=X.dtype)
        X = hstack([X, extra], format="csr")

    return X
