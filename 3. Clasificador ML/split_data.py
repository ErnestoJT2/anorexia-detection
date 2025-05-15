"""
Autor: Ernesto Juarez Torres A01754887  
Fecha: 2025-05

Este script carga el archivo `data_final.csv`, genera la columna 'label' si aún no existe,
y divide los datos aleatoriamente en tres conjuntos: 70 % entrenamiento, 15 % validación
y 15 % prueba. Los índices generados se guardan como archivos `.npy` para referencia posterior.

"""

from pathlib import Path
import pandas as pd
import numpy as np
import re

# Rutas de entrada y salida
ROOT    = Path(__file__).resolve().parent.parent
CSV     = ROOT / "1. Preprocesamiento de Texto" / "data_final.csv"
SPLITS  = Path(__file__).parent / "splits"
SPLITS.mkdir(exist_ok=True)

# ------------------------ Añadir columna 'label' si no existe ------------------------
df = pd.read_csv(CSV)

if "label" not in df.columns:
    if "classe" in df.columns:
        df["label"] = (df["classe"].str.lower().str.strip() != "control").astype(int)
        print(" 'label' creada desde columna 'classe'")
    else:
        df["label"] = df["tweet_text"].str.contains(r"\banorexia\b", flags=re.I).astype(int)
        print(" 'label' creada con keyword 'anorexia'")
    df.to_csv(CSV, index=False)
else:
    print(" 'label' ya existe")

# ------------------------ División aleatoria reproducible ------------------------
N = len(df)
perm = np.random.RandomState(42).permutation(N)
n_train = int(0.70 * N)
n_valid = int(0.15 * N)

train_idx = perm[:n_train]
valid_idx = perm[n_train:n_train + n_valid]
test_idx  = perm[n_train + n_valid:]

# Guardar índices como .npy
for name, arr in zip(("train", "valid", "test"), (train_idx, valid_idx, test_idx)):
    np.save(SPLITS / f"{name}_idx.npy", arr)

print(f" Índices guardados en {SPLITS}  "
      f"train={len(train_idx)}, valid={len(valid_idx)}, test={len(test_idx)}")
