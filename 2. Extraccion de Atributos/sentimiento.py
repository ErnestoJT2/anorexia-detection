"""
sentimiento.py – Análisis de sentimiento para textos en español.

Este módulo utiliza la librería `sentiment_analysis_spanish` para obtener una puntuación
de sentimiento en el rango [0, 1] para cada documento del corpus. Las puntuaciones
más cercanas a 1 indican un sentimiento positivo; más cercanas a 0, negativo.

Autor: Ernesto Juarez Torres A01754887
Fecha: 2025-05
"""

from sentiment_analysis_spanish import sentiment_analysis
import numpy as np

__all__ = ["sentiment_vector"]

# Instancia del modelo de análisis de sentimiento
_sa = sentiment_analysis.SentimentAnalysisSpanish()

def sentiment_vector(corpus):
    """
    Aplica análisis de sentimiento a cada documento del corpus.

    Parámetros:
    - corpus : list[str]
        Lista de documentos en español, uno por entrada.

    Retorna:
    - np.ndarray, shape (n_docs, 1)
        Vector columna con las puntuaciones de sentimiento entre 0 (muy negativo) y 1 (muy positivo).
    """
    return np.array([_sa.sentiment(txt) for txt in corpus], dtype=np.float32).reshape(-1, 1)
