"""
Autor: Ernesto Juárez Torres A01754887

2_normalizacion
===============

Convierte texto a minúsculas; se asume que ya ha sido limpiado de
ruido por `1_eliminacion_de_ruido.py`.

Functions
---------
a_minusculas(texto: str | None) -> str | None
"""

__all__ = ["a_minusculas"]

def a_minusculas(texto: str) -> str:
    """Convierte el texto a minúsculas (ya limpio)."""
    return texto.lower() if texto else texto

if __name__ == "__main__":
    print(a_minusculas("¡HOLA MUNDO!"))
