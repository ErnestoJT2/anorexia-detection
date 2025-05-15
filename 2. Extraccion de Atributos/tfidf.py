"""
Autor: Ernesto Juarez Torres A01754887
Fecha: 2025-05

Este módulo implementa la función `tfidf_features`, que convierte un corpus de texto
en una matriz dispersa de características utilizando la técnica de TF-IDF con unigramas.
Incluye padding si el número de características es menor al máximo definido.

"""

from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix, hstack

__all__ = ["tfidf_features"]

def tfidf_features(corpus, max_features=5000):
    """
    Genera una matriz TF-IDF con unigramas a partir de un corpus textual.

    Parámetros:
    - corpus : list[str]
        Lista de documentos preprocesados, con lemas separados por espacio.
    - max_features : int
        Número máximo de características que se conservarán (las más frecuentes).

    Retorna:
    - scipy.sparse.csr_matrix
        Matriz dispersa de tamaño (n_docs, max_features). Si se encuentran menos
        características, se rellena con columnas de ceros (padding).
    """
    vec = TfidfVectorizer(
        tokenizer=str.split,      # Usa los espacios como separador de tokens
        lowercase=False,          # Se asume que el texto ya está en la forma deseada
        max_features=max_features,
        ngram_range=(1, 1)        # Solo unigramas
    )

    X = vec.fit_transform(corpus)

    # Padding si se generan menos columnas de las deseadas
    n_docs, n_feats = X.shape
    if n_feats < max_features:
        extra = csr_matrix((n_docs, max_features - n_feats), dtype=X.dtype)
        X = hstack([X, extra], format="csr")

    return X
