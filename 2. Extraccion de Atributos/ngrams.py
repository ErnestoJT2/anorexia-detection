"""
Autor: Ernesto Juarez Torres A01754887
Fecha: 2025-05

Este módulo implementa la función `ngram_features`, que transforma un corpus textual 
en una matriz dispersa utilizando la técnica de n-gramas con `CountVectorizer`.
La salida es una representación numérica donde cada fila representa un documento y 
cada columna un n-grama encontrado en el corpus. Se incluye padding si el número 
de características es menor al máximo especificado.

"""

from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import csr_matrix, hstack

__all__ = ["ngram_features"]

def ngram_features(corpus, n=(2, 3), max_features=8000):
    """
    Genera una matriz de características basada en n-gramas (bi-gramas y tri-gramas).

    Parámetros:
    - corpus : list[str]
        Lista de documentos como strings, con lemas separados por espacio.
    - n : tuple(int, int)
        Rango de n-gramas a generar. Por defecto, (2, 3) para bi- y tri-gramas.
    - max_features : int
        Número máximo de características que se conservarán (más frecuentes).

    Retorna:
    - scipy.sparse.csr_matrix
        Matriz dispersa de tamaño (n_docs, max_features). Si se encuentran menos
        n-gramas que `max_features`, se realiza padding con columnas de ceros.
    """
    vec = CountVectorizer(
        tokenizer=str.split,       # Usa espacios como separador de tokens
        lowercase=False,           # No convierte a minúsculas (se espera preprocesado)
        max_features=max_features,
        ngram_range=n              # Rango de n-gramas (ej. bigramas y trigramas)
    )

    X = vec.fit_transform(corpus)

    # Padding si se generan menos columnas de las deseadas
    n_docs, n_feats = X.shape
    if n_feats < max_features:
        extra = csr_matrix((n_docs, max_features - n_feats), dtype=X.dtype)
        X = hstack([X, extra], format="csr")

    return X
