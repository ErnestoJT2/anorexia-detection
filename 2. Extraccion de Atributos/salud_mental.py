"""
Autor: Ernesto Juárez Torres
Fecha: 2025-05

Calcula la frecuencia de aparición de palabras clave relacionadas con TCA,
limpiando tokens para eliminar comillas, tildes y puntuación simple.
"""

import numpy as np
from typing import List
import unidecode

__all__ = ["keywords_matrix", "KEYWORDS"]

KEYWORDS = [
    "comer", "comida", "anorexia", "bulimia", "atracon", "vomitar",
    "fit", "gym", "peso", "adelgazar", "cuerpo", "hambre"
]

def keywords_matrix(corpus: List[str]) -> np.ndarray:
    """
    Calcula la frecuencia de aparición de cada palabra clave por documento,
    usando limpieza de acentos y puntuación básica.

    Parámetros:
    - corpus: Lista de documentos con tokens separados por espacio.

    Retorna:
    - np.ndarray (n_docs, len(KEYWORDS)) con conteo por keyword.
    """
    kw2idx = {kw: i for i, kw in enumerate(KEYWORDS)}
    m = np.zeros((len(corpus), len(KEYWORDS)), dtype=np.int16)

    for i, doc in enumerate(corpus):
        tokens = [unidecode.unidecode(t.strip("',\"")).lower() for t in doc.split()]
        for tok in tokens:
            if tok in kw2idx:
                m[i, kw2idx[tok]] += 1

    return m
