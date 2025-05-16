"""
Autor: Ernesto Juárez Torres A01754887
Fecha: 2025-05

Test para validar la extracción de palabras clave (keywords_matrix)
sobre un corpus mínimo, con validación de dimensiones y conteos.
"""

import sys
from pathlib import Path
import numpy as np

# Añadir la ruta a la carpeta que contiene salud_mental.py
sys.path.append(str(Path(__file__).resolve().parent.parent / "2. Extraccion de Atributos"))

from salud_mental import keywords_matrix, KEYWORDS

def test_keywords_freq():
    tiny_corpus = ["hola mundo comida atracón", "anorexia sin esperanza caminando"]
    M = keywords_matrix(tiny_corpus)

    assert M.shape == (2, len(KEYWORDS)), f"Shape inesperado: {M.shape}, se esperaban {len(KEYWORDS)} columnas"

    comida_idx = KEYWORDS.index("comida")
    atracon_idx = KEYWORDS.index("atracon")
    anorexia_idx = KEYWORDS.index("anorexia")

    assert M[0, comida_idx] == 1, "La palabra 'comida' no fue detectada en el primer documento"
    assert M[0, atracon_idx] == 1, "La palabra 'atracon' no fue detectada en el primer documento"
    assert M[1, anorexia_idx] == 1, "La palabra 'anorexia' no fue detectada en el segundo documento"
