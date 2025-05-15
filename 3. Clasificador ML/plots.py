"""
Autor: Ernesto Juarez Torres A01754887  
Fecha: 2025-05

Este script carga las imágenes PNG generadas por diferentes modelos (Random Forest, SVM, etc.),
las organiza en una única figura horizontal (`roc_all.png`) y la guarda en disco. 
Debe ejecutarse una vez que los modelos han sido entrenados y evaluados.

"""

from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image

# Definición de ruta base y salida
BASE = Path(__file__).parent
OUT  = BASE / "out"

# Nombres esperados de las imágenes ROC generadas por scripts previos
names = ["roc_random_forest.png", "roc_svm.png", "roc_svm_balanced.png"]

# Cargar imágenes existentes
imgs = [Image.open(OUT / n) for n in names if (OUT / n).exists()]
if not imgs:
    raise SystemExit("No se encontraron ROC png. Ejecuta antes los scripts de entrenamiento.")

# Configurar figura para contener todas las imágenes horizontalmente
cols = len(imgs)
fig, axes = plt.subplots(1, cols, figsize=(4 * cols, 4))

if cols == 1:
    axes = [axes]  # Garantiza que axes sea iterable si solo hay una imagen

# Mostrar cada imagen en un subplot con su título
for ax, img, title in zip(axes, imgs, ["RF", "SVM", "SVM_bal"][:cols]):
    ax.imshow(img)
    ax.axis("off")
    ax.set_title(title)

plt.tight_layout()
plt.savefig(OUT / "roc_all.png", dpi=140)
print("roc_all.png guardado en", OUT)
