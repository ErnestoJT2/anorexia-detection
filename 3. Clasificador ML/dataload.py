# dataload.py
from pathlib import Path
import pandas as pd
import numpy as np

def load_all_features_and_labels():
    """
    Carga matrices de atributos desde carpeta 2 y etiquetas desde textos_originales.csv.
    """
    base_feat = Path(__file__).resolve().parent.parent / "2. Extraccion de Atributos" / "out"
    base_raw  = base_feat  # ahora está en la misma carpeta

    # === Cargar matrices de atributos ===
    tfidf      = pd.read_csv(base_feat / "tfidf.csv")
    bow        = pd.read_csv(base_feat / "bow.csv")
    ngrams     = pd.read_csv(base_feat / "ngrams.csv")
    keywords   = pd.read_csv(base_feat / "keywords.csv")
    sentiment  = pd.read_csv(base_feat / "sentiment.csv")
    stylistic  = pd.read_csv(base_feat / "stylistic.csv")

    # === Cargar etiquetas desde textos_originales.csv ===
    df_raw = pd.read_csv(base_raw / "textos_originales.csv")

    if "class" not in df_raw.columns:
        raise ValueError("La columna 'class' no fue encontrada en textos_originales.csv")

    y = df_raw["class"].astype(str).str.lower().str.strip().map({
        "control": 0,
        "anorexia": 1
    }).values

    if pd.isnull(y).any():
        raise ValueError("Se encontraron etiquetas no válidas en la columna 'class'")

    # === Combinar atributos ===
    X = pd.concat([tfidf, bow, ngrams, keywords, sentiment, stylistic], axis=1)

    return X, y
