"""
Autor: Ernesto Juarez Torres A01754887  
Fecha: 2025-05

Este servicio expone un endpoint `/predict` que recibe texto y devuelve la predicción
(realizada por un modelo LogisticRegression entrenado con TF-IDF bigramas).
Además, sirve una interfaz HTML y abre automáticamente el navegador.

"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel  # type: ignore

import webbrowser
import threading
import time

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# ---------------------- Inicialización ----------------------
app = FastAPI()

class PredictionRequest(BaseModel):
    text: str  # Entrada del usuario para predicción

# ---------------------- Carga y entrenamiento ----------------------
# Cargar dataset preprocesado
data = pd.read_csv(r'C:\Users\ernes\OneDrive\Escritorio\Reto Final\1. Preprocesamiento de Texto\data_final.csv')

# Limpieza rápida de campo
data['processed_text'] = data['tweet_text'].str.replace("[\[\]',]", "", regex=True)

# Vectorización con bigramas
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
X_tfidf = tfidf_vectorizer.fit_transform(data['processed_text'])

# Entrenamiento del modelo
model = LogisticRegression(max_iter=500, random_state=42)
model.fit(X_tfidf, data['class'].map({'control': 0, 'anorexia': 1}))

# ---------------------- Endpoint de predicción ----------------------
@app.post('/predict')
def predict(request: PredictionRequest):
    """
    Realiza una predicción sobre el texto recibido.
    Retorna clase ('anorexia' o 'control') y probabilidad.
    """
    text_tfidf = tfidf_vectorizer.transform([request.text])
    prediction = model.predict(text_tfidf)[0]
    probability = model.predict_proba(text_tfidf)[0][1]

    return {
        'prediction': 'anorexia' if prediction == 1 else 'control',
        'probability': probability
    }

# ---------------------- Servir HTML y abrir navegador ----------------------
app.mount("/", StaticFiles(directory=".", html=True), name="static")

def open_browser():
    time.sleep(2)  # Espera breve para asegurar que el servidor esté activo
    webbrowser.open("http://127.0.0.1:8000/anorexia.html")

# Ejecutar apertura en segundo plano
threading.Thread(target=open_browser).start()
