"""
Autor: Ernesto Juarez Torres A01754887
Fecha: 2025-05

Este script entrena un modelo base de clasificación usando Random Forest para la detección
de publicaciones con señales de anorexia. Utiliza los conjuntos de entrenamiento y validación
generados previamente, evalúa el modelo con métricas estándar (AUC-ROC y F1), y guarda:

- Un reporte de clasificación (reporte_rf_baseline.txt)
- Las métricas clave en formato CSV (metrics.csv)
- La curva ROC como imagen (roc_rf_baseline.png)

Este modelo sirve como referencia para comparar contra versiones optimizadas.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, f1_score, RocCurveDisplay

# === Cargar datos ===
BASE = Path(__file__).parent
OUT = BASE / "out"
train_df = pd.read_csv(OUT / "train.csv")
val_df = pd.read_csv(OUT / "val.csv")

X_train = train_df.drop(columns=["class"])
y_train = train_df["class"]
X_val = val_df.drop(columns=["class"])
y_val = val_df["class"]

# === Modelo base: Random Forest ===
clf = RandomForestClassifier(random_state=42, n_jobs=-1)
clf.fit(X_train, y_train)

pred = clf.predict(X_val)
proba = clf.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, proba)
f1 = f1_score(y_val, pred)

# === Guardar reporte ===
reporte = classification_report(y_val, pred, target_names=["control", "anorexia"])
with open(OUT / "reporte_rf_baseline.txt", "w") as f:
    f.write(reporte)

# === Guardar métricas ===
pd.DataFrame([{"modelo": "rf_baseline", "AUC": auc, "F1": f1}]) \
  .to_csv(OUT / "metrics.csv", index=False)

# === Guardar curva ROC ===
RocCurveDisplay.from_predictions(y_val, proba).plot()
plt.title("ROC – RF Baseline")
plt.tight_layout()
plt.savefig(OUT / "roc_rf_baseline.png")
plt.close()

# === Imprimir resumen ===
print("✅ Random Forest baseline entrenado")
print(f"🔹 AUC: {auc:.3f} | F1: {f1:.3f}")
print("🔍 Reporte de clasificación:\n")
print(reporte)
print