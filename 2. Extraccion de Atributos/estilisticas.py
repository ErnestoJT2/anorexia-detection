"""
Autor: Ernesto Juarez Torres A01754887
Fecha: 2025-05

Este módulo proporciona la función `stylistic_matrix`, que calcula tres atributos
estilísticos por documento: longitud en tokens, número de pronombres personales,
y longitud promedio de oración (aproximada por la cantidad de tokens, ya que no se
utiliza puntuación).

"""

import numpy as np

__all__ = ["stylistic_matrix"]

def stylistic_matrix(corpus):
    """
    Calcula métricas estilísticas básicas para cada documento del corpus.

    Parámetros:
    - corpus : list[str]
        Lista de documentos preprocesados, donde cada documento es un string de tokens separados por espacios.

    Retorna:
    - np.ndarray, shape (n_docs, 3)
        Matriz con tres columnas por documento:
        [0] Número total de tokens.
        [1] Número de pronombres personales encontrados.
        [2] Longitud promedio de oración (aproximada por cantidad de tokens, ya que no hay puntuación).
    """
    # Lista de pronombres personales en español (forma simplificada)
    pronouns = {"yo", "tú", "vos", "él", "ella", "nosotros", "vosotros", "ustedes", "ellos", "ellas", "me", "te", "se"}

    # Inicializa matriz de salida
    m = np.zeros((len(corpus), 3), dtype=np.float32)

    for i, doc in enumerate(corpus):
        tokens = doc.split()
        m[i, 0] = len(tokens)                                 # Total de tokens
        m[i, 1] = sum(tok in pronouns for tok in tokens)      # Conteo de pronombres personales
        m[i, 2] = len(tokens)                                 # Longitud de oración ≈ tokens (sin puntuación)

    return m
