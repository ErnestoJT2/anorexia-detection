"""
pipeline_features.py
Autor: Ernesto Juárez Torres A01754887
Fecha: 2025-05

Pipeline completo para extracción de atributos, conversión de clases ("anorexia", "control")
a valores numéricos, división 70/15/15, y guardado de archivos y gráficos.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from sklearn.model_selection import train_test_split

from bow import bow_features
from tfidf import tfidf_features
from ngrams import ngram_features
from salud_mental import keywords_matrix, KEYWORDS
from sentimiento import sentiment_vector
from estilisticas import stylistic_matrix

# === Configuración de pandas y warnings ===
pd.options.display.float_format = '{:,.2f}'.format
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')

# === Rutas ===
BASE = Path(__file__).parent
PREPROC = BASE.parent / "1. Preprocesamiento de Texto" / "data_final.csv"
OUT = BASE / "out"
OUT.mkdir(exist_ok=True)

# === Carga corpus y clase ===
df = pd.read_csv(PREPROC)
corpus = df["tweet_text"].astype(str).tolist()

if "class" not in df.columns:
    raise ValueError("No se encontró la columna 'class' en el archivo CSV.")

# Mapeo string → número
y_str = df["class"].astype(str).str.lower().str.strip()
y = y_str.map({"control": 0, "anorexia": 1})

if y.isnull().any():
    raise ValueError("Se encontraron valores no válidos en 'class' (solo se aceptan 'control' o 'anorexia').")

# === 1. TF-IDF ===
tfidf_df = pd.DataFrame(tfidf_features(corpus, max_features=5000).toarray())
tfidf_df.to_csv(OUT / "tfidf.csv", index=False)
print(f"TF-IDF generado: {tfidf_df.shape}")
tfidf_df.sum().sort_values(ascending=False).head(20).plot(kind="bar", title="Top 20 términos TF-IDF")
plt.tight_layout()
plt.savefig(OUT / "grafico_tfidf.png")
plt.clf()

# === 2. BoW ===
bow_df = pd.DataFrame(bow_features(corpus, max_features=5000).toarray())
bow_df.to_csv(OUT / "bow.csv", index=False)
print(f"BoW generado: {bow_df.shape}")
bow_df.sum().sort_values(ascending=False).head(20).plot(kind="bar", title="Top 20 términos BoW")
plt.tight_layout()
plt.savefig(OUT / "grafico_bow.png")
plt.clf()

# === 3. N-grams ===
ngram_df = pd.DataFrame(ngram_features(corpus, n=(2, 3), max_features=8000).toarray())
ngram_df.to_csv(OUT / "ngrams.csv", index=False)
print(f"N-grams generado: {ngram_df.shape}")
ngram_df.sum().sort_values(ascending=False).head(20).plot(kind="bar", title="Top 20 n-gramas")
plt.tight_layout()
plt.savefig(OUT / "grafico_ngrams.png")
plt.clf()

# === 4. Keywords ===
kw_df = pd.DataFrame(keywords_matrix(corpus), columns=KEYWORDS)
kw_df.round(2).to_csv(OUT / "keywords.csv", index=False)
print("Keywords:")
print(kw_df.describe().round(2))
sns.heatmap(kw_df.mean().to_frame().T, cmap="YlGnBu", annot=True, cbar=False)
plt.title("Promedio de activación por palabra clave")
plt.tight_layout()
plt.savefig(OUT / "grafico_keywords.png")
plt.clf()

# === 5. Sentimiento ===
sent_df = pd.DataFrame(sentiment_vector(corpus), columns=["sentimiento"])
sent_df.round(2).to_csv(OUT / "sentiment.csv", index=False)
print("Sentimiento:")
print(sent_df.describe().round(2))
sent_df.plot(kind="hist", bins=15, title="Distribución del sentimiento", figsize=(8, 5))
plt.tight_layout()
plt.savefig(OUT / "grafico_sentimiento.png")
plt.clf()

# === 6. Estilo ===
sty_df = pd.DataFrame(stylistic_matrix(corpus), columns=["n_tokens", "pronombres", "long_oracion"])
sty_df.round(2).to_csv(OUT / "stylistic.csv", index=False)
print("Métricas estilísticas:")
print(sty_df.describe().round(2))
sns.boxplot(data=sty_df)
plt.title("Distribución de métricas estilísticas")
plt.tight_layout()
plt.savefig(OUT / "grafico_estilo.png")
plt.clf()

# === 7. Combinación final y división ===
print("\n Combinando todos los atributos y dividiendo en 70/15/15...")

# === 8. Guardar texto original y etiqueta ===
df_texto = df[["tweet_text", "class"]].copy()
df_texto.to_csv(OUT / "textos_originales.csv", index=False)
print(f"\n📄 Archivo 'textos_originales.csv' guardado con {df_texto.shape[0]} ejemplos.")

X = pd.concat([tfidf_df, bow_df, ngram_df, kw_df, sent_df, sty_df], axis=1)
X["class"] = y.values

X_train, X_temp = train_test_split(X, test_size=0.30, stratify=X["class"], random_state=42)
X_val, X_test = train_test_split(X_temp, test_size=0.5, stratify=X_temp["class"], random_state=42)

X.to_csv(OUT / "features_final.csv", index=False)
X_train.to_csv(OUT / "train.csv", index=False)
X_val.to_csv(OUT / "val.csv", index=False)
X_test.to_csv(OUT / "test.csv", index=False)

print("División completada:")
print(f"  Entrenamiento: {X_train.shape}")
print(f"  Validación:    {X_val.shape}")
print(f"  Prueba:        {X_test.shape}")
