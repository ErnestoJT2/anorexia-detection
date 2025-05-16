# plot_metrics.py
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# === Cargar métricas ===
BASE = Path(__file__).parent
OUT = BASE / "out"
OUT.mkdir(exist_ok=True)

metrics_file = OUT / "metrics.csv"
if not metrics_file.exists():
    print("No se encontró metrics.csv. Ejecuta primero los archivos de entrenamiento.")
    exit(1)

df = pd.read_csv(metrics_file).drop_duplicates(subset="modelo", keep="last")

# === Generar heatmap ===
plt.figure(figsize=(5, 3))
sns.heatmap(df.set_index("modelo"), annot=True, cmap="Blues", fmt=".3f")
plt.title("Comparación de modelos (AUC y F1)")
plt.tight_layout()
plt.savefig(OUT / "comparacion_heatmap.png")
plt.close()

print("\n✅ Comparación generada:")
print(df.round(3).to_string(index=False))