"""
Autor: Ernesto Juarez Torres A01754887
Fecha: 2025-05

Este módulo carga el corpus preprocesado, genera representaciones TF-IDF, Bag of Words,
N-gramas, y matrices de características específicas (keywords, sentimiento y estilo).
Los resultados se guardan en formatos comprimidos (.npz, .npy, .joblib) para su uso posterior.

"""

from pathlib import Path
import pandas as pd
import numpy as np
import scipy.sparse as sp
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from salud_mental import keywords_matrix
from sentimiento import sentiment_vector
from estilisticas import stylistic_matrix

# Rutas
BASE     = Path(__file__).parent
PREPROC  = BASE.parent / "1. Preprocesamiento de Texto" / "data_final.csv"
OUT      = BASE / "out"
OUT.mkdir(exist_ok=True)

# Lectura del corpus (lista de strings)
df     = pd.read_csv(PREPROC)
corpus = df["tweet_text"].tolist()

# 1) TF-IDF
tfidf_vect = TfidfVectorizer()
tfidf_mat  = tfidf_vect.fit_transform(corpus)
joblib.dump(tfidf_vect, OUT / "tfidf_vect.joblib")
sp.save_npz(OUT / "tfidf.npz", tfidf_mat)

# 2) Bag of Words
bow_vect = CountVectorizer()
bow_mat  = bow_vect.fit_transform(corpus)
joblib.dump(bow_vect, OUT / "bow_vect.joblib")
sp.save_npz(OUT / "bow.npz", bow_mat)

# 3) N-grams (bi- y tri-gramas)
ngram_vect = CountVectorizer(ngram_range=(2,3))
ngram_mat  = ngram_vect.fit_transform(corpus)
joblib.dump(ngram_vect, OUT / "ngram_vect.joblib")
sp.save_npz(OUT / "ngrams.npz", ngram_mat)

# 4) Frecuencia de palabras clave
kw = keywords_matrix(corpus)
np.save(OUT / "keywords.npy", kw)

# 5) Sentimiento
sent = sentiment_vector(corpus)
np.save(OUT / "sentiment.npy", sent)

# 6) Métricas estilísticas
sty = stylistic_matrix(corpus)
np.save(OUT / "stylistic.npy", sty)

print("Atributos y vectorizadores guardados en", OUT)

# === EXPORTABLE PARA TEST ===
def tfidf_features(corpus, max_features=1000):
    vect = TfidfVectorizer(max_features=max_features)
    return vect.fit_transform(corpus)

def main():
    print("Ejecución completa de pipeline_features.py")
    test_corpus = ["hola mundo comida atracón", "anorexia sin esperanza caminando"]
    tfidf = tfidf_features(test_corpus, max_features=20)
    sp.save_npz(OUT / "tfidf.npz", tfidf)
    kw = keywords_matrix(test_corpus)
    np.save(OUT / "keywords.npy", kw)