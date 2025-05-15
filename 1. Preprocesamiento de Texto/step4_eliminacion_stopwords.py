"""
Autor: Ernesto Juárez Torres A01754887

4_eliminacion_stopwords
=======================

Filtra stop‑words en español usando NLTK, preservando términos de
interés para TCA (Trastornos de la Conducta Alimentaria).

Functions
---------
eliminar_stopwords(tokens: list[str]) -> list[str]
"""

from nltk.corpus import stopwords

__all__ = ["eliminar_stopwords"]

_STOP_ES = set(stopwords.words("spanish"))
_TERMINOS_RELEVANTES = {
    "anorexia", "bulimia", "atracón", "evitación", "restricción", "ingesta"
}
_STOP_CUSTOM = _STOP_ES - _TERMINOS_RELEVANTES

def eliminar_stopwords(tokens):
    """Filtra las stop‑words salvo las relevantes para salud mental."""
    return [t for t in tokens if t.lower() not in _STOP_CUSTOM]

if __name__ == "__main__":
    print(eliminar_stopwords(["la", "anorexia", "es", "seria"]))
