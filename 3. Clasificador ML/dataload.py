"""
Autor: Ernesto Juarez Torres A01754887  
Fecha: 2025-05

Este módulo carga las matrices de características previamente generadas en la carpeta
`2/out`, las concatena horizontalmente y asocia las etiquetas (`y`) extraídas desde el
archivo `data_final.csv`. Es útil como entrada para modelos supervisados.

"""

from pathlib import Path
import numpy as np
import scipy.sparse as sp
import pandas as pd

# Rutas base
ROOT = Path(__file__).resolve().parent.parent
OUT2 = ROOT / "2. Extraccion de Atributos" / "out"

def load_all_matrices():
    """
    Carga todas las matrices de características generadas (sparse y dense),
    concatena en una sola matriz `X` y asocia las etiquetas `y`.

    Retorna:
    - X : scipy.sparse.csr_matrix
        Matriz dispersa de características combinadas.
    - y : np.ndarray
        Vector con las etiquetas binarias del corpus (0=control, 1=anorexia).
    """

    # 1) Carga de matrices dispersas y densas
    mats = [
        sp.load_npz(OUT2 / "tfidf.npz"),
        sp.load_npz(OUT2 / "ngrams.npz"),
        sp.load_npz(OUT2 / "bow.npz"),
        sp.csr_matrix(np.load(OUT2 / "keywords.npy")),
        sp.csr_matrix(np.load(OUT2 / "sentiment.npy")),
        sp.csr_matrix(np.load(OUT2 / "stylistic.npy")),
    ]
    X = sp.hstack(mats).tocsr()

    # 2) Carga etiquetas desde el CSV preprocesado
    CSV = ROOT / "1. Preprocesamiento de Texto" / "data_final.csv"
    df = pd.read_csv(CSV)

    # Si no existe la columna 'label', la generamos
    if "label" not in df.columns:
        if "classe" in df.columns:
            # Asume 'classe' con 'control' y otros
            df["label"] = (df["classe"].str.lower() != "control").astype(int)
        else:
            # Busca palabra clave 'anorexia' en el texto como fallback
            df["label"] = (
                df["tweet_text"]
                  .str.contains(r"\banorexia\b", case=False, na=False)
                  .astype(int)
            )

    y = df["label"].values
    return X, y
