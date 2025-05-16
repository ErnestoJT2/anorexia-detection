"""
Autor: Ernesto Juárez Torres A01754887
Fecha: 2025-05

Este módulo carga las matrices de atributos generadas en la etapa de extracción
(TF-IDF, BoW, N-grams, keywords, sentimiento y estilo), así como las etiquetas desde
el archivo textos_originales.csv. Realiza validaciones y devuelve las matrices
X (atributos) y y (etiquetas) listas para el entrenamiento.

Este archivo forma parte del sistema de evaluación de modelos para detección
automática de publicaciones con riesgo de anorexia, siguiendo el protocolo del curso TC3002B.
"""

from pathlib import Path
import pandas as pd
import numpy as np

def load_all_features_and_labels():
    """
    Carga matrices de atributos desde carpeta 2 y etiquetas desde textos_originales.csv.
    Devuelve:
        X (pd.DataFrame): Matriz combinada de atributos lingüísticos.
        y (np.ndarray): Vector de etiquetas binario (0=control, 1=anorexia).
    """
    base_feat = Path(__file__).resolve().parent.parent / "2. Extraccion de Atributos" / "out"

    # Cargar matrices de atributos
    tfidf     = pd.read_csv(base_feat / "tfidf.csv")
    bow       = pd.read_csv(base_feat / "bow.csv")
    ngrams    = pd.read_csv(base_feat / "ngrams.csv")
    keywords  = pd.read_csv(base_feat / "keywords.csv")
    sentiment = pd.read_csv(base_feat / "sentiment.csv")
    stylistic = pd.read_csv(base_feat / "stylistic.csv")

    # Cargar etiquetas desde textos_originales.csv
    df_raw = pd.read_csv(base_feat / "textos_originales.csv")

    if "class" not in df_raw.columns:
        raise ValueError("La columna 'class' no fue encontrada en textos_originales.csv")

    y = df_raw["class"].astype(str).str.lower().str.strip().map({
        "control": 0,
        "anorexia": 1
    }).values

    if pd.isnull(y).any():
        raise ValueError("Se encontraron etiquetas no válidas en la columna 'class'")

    # Concatenar todas las matrices horizontalmente
    X = pd.concat([tfidf, bow, ngrams, keywords, sentiment, stylistic], axis=1)

    return X, y
