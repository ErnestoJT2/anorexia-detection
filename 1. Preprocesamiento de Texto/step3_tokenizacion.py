"""
Autor: Ernesto Juárez Torres A01754887

3_tokenizacion
==============

Tokeniza texto (ya limpio y en minúsculas) usando NLTK.

Functions
---------
tokenizar_texto(texto: str) -> list[str]
"""

import nltk
from nltk.tokenize import word_tokenize

__all__ = ["tokenizar_texto"]

# Comprueba que 'punkt' esté disponible; si no, lo descarga.
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

def tokenizar_texto(texto: str):
    """
    Tokeniza el texto en palabras usando NLTK.
    Se especifica 'spanish' para que maneje mejor signos '¿' '¡', etc.
    """
    return word_tokenize(texto, language="spanish")
