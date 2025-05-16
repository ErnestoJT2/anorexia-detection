"""
Autor: Ernesto Juarez Torres A01754887
Fecha: 2025-05

Este módulo realiza la división del conjunto completo de datos en tres subconjuntos:
entrenamiento (70 %), validación (15 %) y prueba (15 %), usando particiones estratificadas
para preservar la proporción entre clases ("anorexia" y "control").

El conjunto de datos se carga mediante `load_all_features_and_labels()` y se guarda en
archivos .csv separados para cada subconjunto en la carpeta /out.

Este paso es fundamental para evaluar el rendimiento general del sistema con datos no vistos.
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from dataload import load_all_features_and_labels

# === Configuración ===
BASE = Path(__file__).parent
OUT = BASE / "out"
OUT.mkdir(exist_ok=True)

# === Cargar atributos y etiquetas ===
X, y = load_all_features_and_labels()
X["class"] = y  # Añadir columna de etiqueta al final

# === División 70% entrenamiento / 30% restante ===
X_train, X_temp = train_test_split(
    X, test_size=0.30, stratify=X["class"], random_state=42
)

# === División del 30% en validación (15%) y prueba (15%) ===
X_val, X_test = train_test_split(
    X_temp, test_size=0.5, stratify=X_temp["class"], random_state=42
)

# === Guardar archivos ===
X_train.to_csv(OUT / "train.csv", index=False)
X_val.to_csv(OUT / "val.csv", index=False)
X_test.to_csv(OUT / "test.csv", index=False)

# === Impresión en consola ===
print("✅ División completada y archivos guardados en /out")
print(f"🔹 Entrenamiento: {X_train.shape[0]} ejemplos "
      f"(anorexia: {(X_train['class']==1).sum()}, control: {(X_train['class']==0).sum()})")
print(f"🔹 Validación:    {X_val.shape[0]} ejemplos "
      f"(anorexia: {(X_val['class']==1).sum()}, control: {(X_val['class']==0).sum()})")
print(f"🔹 Prueba:        {X_test.shape[0]} ejemplos "
      f"(anorexia: {(X_test['class']==1).sum()}, control: {(X_test['class']==0).sum()})")
