"""
Autor: Ernesto Juárez Torres
Fecha: 2025-05

Calcula métricas estilísticas simples: número de tokens, pronombres personales y longitud aproximada.
"""

import numpy as np
from typing import List

__all__ = ["stylistic_matrix"]

def stylistic_matrix(corpus: List[str]) -> np.ndarray:
    """
    Extrae métricas estilísticas por documento.

    Parámetros:
    - corpus: Lista de strings con tokens separados por espacio.

    Retorna:
    - np.ndarray (n_docs, 3):
        [0] Total de tokens
        [1] Pronombres personales
        [2] Longitud aproximada (n_tokens)
    """
    pronombres = {"yo", "tú", "vos", "él", "ella", "nosotros", "vosotros", "ustedes", "ellos", "ellas", "me", "te", "se"}
    m = np.zeros((len(corpus), 3), dtype=np.float32)

    for i, doc in enumerate(corpus):
        tokens = doc.split()
        m[i, 0] = len(tokens)
        m[i, 1] = sum(tok in pronombres for tok in tokens)
        m[i, 2] = len(tokens)  # Equivalente a longitud aproximada

    return m
