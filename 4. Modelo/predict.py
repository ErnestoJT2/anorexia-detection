"""
Autor: Ernesto Juarez Torres A01754887
Fecha: 2025-05

Servicio FastAPI que expone un endpoint `/predict` para clasificar texto como 'anorexia' o 'control'.
Se entrena dinámicamente desde textos_originales.csv con TF-IDF bigramas y RandomForest.
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import webbrowser
import threading
import time

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path

# ---------------------- Inicialización ----------------------
app = FastAPI()

class PredictionRequest(BaseModel):
    text: str

# ---------------------- Entrenamiento dinámico ----------------------
BASE = Path(__file__).resolve().parent
CSV = BASE.parent / "2. Extraccion de Atributos" / "out" / "textos_originales.csv"

# Cargar corpus y etiquetas
df = pd.read_csv(CSV)
corpus = df["tweet_text"].astype(str).tolist()
y = df["class"].astype(str).str.lower().str.strip().map({"control": 0, "anorexia": 1}).values

# Vectorización TF-IDF bigramas
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
X = tfidf_vectorizer.fit_transform(corpus)

# Entrenar modelo rápido
model = RandomForestClassifier(random_state=42)
model.fit(X, y)

print("✅ Modelo y vectorizador entrenados desde textos_originales.csv")

# ---------------------- Endpoint de predicción ----------------------
@app.post("/predict")
def predict(request: PredictionRequest):
    vector = tfidf_vectorizer.transform([request.text])
    pred = model.predict(vector)[0]
    prob = model.predict_proba(vector)[0][1]

    return {
        "prediction": "anorexia" if pred == 1 else "control",
        "probability": round(prob, 3)
    }

# ---------------------- Interfaz web ----------------------
app.mount("/", StaticFiles(directory=".", html=True), name="static")

def open_browser():
    time.sleep(1)
    webbrowser.open("http://127.0.0.1:8000/anorexia.html")

threading.Thread(target=open_browser).start()
