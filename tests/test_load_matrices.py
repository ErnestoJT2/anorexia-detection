#Autor: Ernesto Juárez Torres A01754887
# tests/test_load_matrices.py
"""
Verifica que las matrices de características y etiquetas se cargan correctamente desde CSV.
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent / "3. Clasificador ML"))
from dataload import load_all_features_and_labels

def test_matrices_dimensions():
    X, y = load_all_features_and_labels()
    assert X.shape[0] == len(y)
    assert X.shape[1] > 1000  # Se espera gran número de atributos
