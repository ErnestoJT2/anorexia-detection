"""
Autor: Ernesto Juárez Torres A01754887

6_abreviaturas
==============

Expande jerga y abreviaturas frecuentes en redes sociales.

Functions
---------
expandir_abreviaturas(tokens: list[str]) -> list[str]
"""

__all__ = ["expandir_abreviaturas"]

_ABREV = {
    "u": "you", "pls": "please", "idk": "i don't know",
    "btw": "by the way", "lol": "laugh out loud",
    # español
    "xq": "porque", "tqm": "te quiero mucho"
}

def expandir_abreviaturas(tokens):
    """Reemplaza tokens según diccionario de jerga/abreviaturas."""
    return [_ABREV.get(t, t) for t in tokens]

if __name__ == "__main__":
    print(expandir_abreviaturas(["pls", "ayuda", "xq"]))
