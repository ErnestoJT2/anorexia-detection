"""
Autor: Ernesto Juárez Torres
Fecha: 2025-05

Devuelve puntuación de sentimiento para cada texto en español [0, 1].
"""

from sentiment_analysis_spanish import sentiment_analysis
import numpy as np
from typing import List

__all__ = ["sentiment_vector"]

_sa = sentiment_analysis.SentimentAnalysisSpanish()

def sentiment_vector(corpus: List[str]) -> np.ndarray:
    """
    Evalúa sentimiento de cada documento.

    Parámetros:
    - corpus: Lista de textos en español.

    Retorna:
    - np.ndarray (n_docs, 1) con valores entre 0 (negativo) y 1 (positivo).
    """
    return np.array([_sa.sentiment(text) for text in corpus], dtype=np.float32).reshape(-1, 1)
