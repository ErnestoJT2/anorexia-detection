"""
Autor: Ernesto Juárez Torres

Este módulo implementa la función `bow_features`, que transforma un corpus textual
en una matriz dispersa usando la técnica Bag of Words (BoW), basada en unigramas.
Además, asegura que la matriz de salida tenga siempre un número fijo de columnas 
rellenando con ceros si es necesario.

Fecha: 2025-05
"""

from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix, hstack

__all__ = ["bow_features"]

def bow_features(corpus, max_features=5000):
    """
    Genera una matriz de características BoW a partir de un corpus de documentos.

    Parámetros:
    - corpus : list[str]
        Lista de strings, donde cada string representa un documento con lemas separados por espacio.
    - max_features : int
        Número máximo de características (palabras) a considerar.

    Retorna:
    - scipy.sparse.csr_matrix
        Matriz dispersa de tamaño (n_docs, max_features), donde cada fila representa un documento
        y cada columna una palabra (unigrama). Si el vocabulario encontrado es menor que `max_features`,
        se realiza un padding con columnas de ceros para mantener la forma fija.
    """
    vec = CountVectorizer(
        tokenizer=str.split,      # Usa los espacios como separador de tokens
        lowercase=False,          # No modifica el texto (ya se espera preprocesado)
        max_features=max_features
    )
    X = vec.fit_transform(corpus)

    # Padding si hay menos características que max_features
    n_docs, n_feats = X.shape
    if n_feats < max_features:
        extra = csr_matrix((n_docs, max_features - n_feats), dtype=X.dtype)
        X = hstack([X, extra], format="csr")

    return X
