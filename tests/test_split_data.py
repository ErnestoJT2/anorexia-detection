# tests/test_split_data.py
"""
Autor: Ernesto Juárez Torres A01754887
Fecha: 2025-05

Verifica que los archivos train.csv, val.csv y test.csv fueron creados correctamente
y que sus proporciones son aproximadamente 70/15/15.
"""

from pathlib import Path
import pandas as pd

def test_split_proporcion():
    base = Path(__file__).resolve().parent.parent / "2. Extraccion de Atributos" / "out"

    df_train = pd.read_csv(base / "train.csv")
    df_val   = pd.read_csv(base / "val.csv")
    df_test  = pd.read_csv(base / "test.csv")

    total = len(df_train) + len(df_val) + len(df_test)
    p_train = len(df_train) / total
    p_val   = len(df_val) / total
    p_test  = len(df_test) / total

    assert abs(p_train - 0.70) < 0.02
    assert abs(p_val   - 0.15) < 0.02
    assert abs(p_test  - 0.15) < 0.02
