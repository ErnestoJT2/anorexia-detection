"""
Autor: Ernesto Juarez Torres A01754887
Fecha: 2025-05

Este módulo proporciona la función `keywords_matrix`, que cuenta la frecuencia de aparición
de un conjunto definido de palabras clave asociadas a trastornos de la conducta alimentaria (TCA)
dentro de cada documento del corpus.

"""

import numpy as np

__all__ = ["keywords_matrix"]

# Lista de palabras clave relacionadas con desórdenes alimenticios
KEYWORDS = ["comida", "atracón", "bulimia", "anorexia", "ingesta", "restricción"]

def keywords_matrix(corpus):
    """
    Calcula la frecuencia de aparición de cada palabra clave en cada documento del corpus.

    Parámetros:
    - corpus : list[str]
        Lista de documentos preprocesados, donde cada documento es un string con tokens separados por espacio.

    Retorna:
    - np.ndarray, shape (n_docs, len(KEYWORDS))
        Matriz densa con el conteo de cada palabra clave por documento.
    """
    m = np.zeros((len(corpus), len(KEYWORDS)), dtype=np.int16)
    kw2idx = {k: i for i, k in enumerate(KEYWORDS)}

    for i, doc in enumerate(corpus):
        for tok in doc.split():
            if tok in kw2idx:
                m[i, kw2idx[tok]] += 1

    return m
