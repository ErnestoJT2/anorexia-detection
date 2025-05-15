"""
Autor: Ernesto Juárez Torres A01754887

5_lematizacion
==============

Reduce cada token a su lema usando `spaCy` (modelo `es_core_news_sm`).

Functions
---------
lematizar_tokens(tokens: list[str]) -> list[str]
"""

import spacy

__all__ = ["lematizar_tokens"]

_nlp = spacy.load("es_core_news_sm", disable=["parser", "ner"])

def lematizar_tokens(tokens):
    """Lematiza y devuelve lista de lemas."""
    text = " ".join(tokens)
    return [tok.lemma_ for tok in _nlp(text) if tok.lemma_.strip()]

if __name__ == "__main__":
    print(lematizar_tokens(["caminando", "comiendo"]))
