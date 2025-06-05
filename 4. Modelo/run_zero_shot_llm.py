"""
Zero-shot suicidal-detection with local Llama-3-8B-Instruct (Q4 GGUF, Ollama)
Autor: Ernesto Juárez Torres
Fase 3 – Evidencia 3 (TC3002B)
"""
import os, json, time, tqdm, requests
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from dotenv import load_dotenv

load_dotenv()                                         # .env con OPENAI_BASE_URL y OPENAI_API_KEY
OLLAMA_URL = os.getenv("OPENAI_BASE_URL", "http://localhost:11434") + "/v1/chat/completions"
MODEL = "llama3:8b"                                   # etiqueta de Ollama
HEADERS = {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY','ollama')}"}

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    texts = df["tweet_text"].tolist()
    labels = (df["class"].str.strip().str.lower() == "anorexia").astype(int).tolist()
    return texts, labels

def llama_classify(texts, temperature=0.0, max_tokens=1):
    """
    Devuelve predicciones binarias y probabilidad suicida (basada en logprobs del token 'SI')
    """
    preds, probs = [], []
    for txt in tqdm.tqdm(texts, desc="LLM-inference"):
        prompt = (
            "Eres un modelo experto en detección de suicidalidad.\n"
            "Lee el siguiente texto y responde únicamente 'SI' si el autor muestra intenciones "
            "explícitas de quitarse la vida, de lo contrario responde 'NO'.\n\n"
            f"Texto: \"{txt}\"\nRespuesta:"
        )
        payload = {
            "model": MODEL,
            "messages":  [
                {"role": "user", "content": prompt}
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "logprobs": True
        }
        r = requests.post(OLLAMA_URL, headers=HEADERS, json=payload, timeout=120)
        r.raise_for_status()
        choice = r.json()["choices"][0]
        response_text = choice["message"]["content"].strip().upper()
        # calcular probabilidad aproximada con logprobs del primer token
        logprob_dict = choice.get("logprobs", {}).get("content", [{}])[0]
        p_yes = np.exp(logprob_dict.get("logprob", -10)) if response_text.startswith("SI") else \
                1 - np.exp(logprob_dict.get("logprob", -10))
        preds.append(1 if response_text.startswith("SI") else 0)
        probs.append(p_yes)
    return preds, probs

def metrics(y_true, y_pred, y_proba):
    return {
        "auc":  roc_auc_score(y_true, y_proba),
        "f1":   f1_score(y_true, y_pred, average="macro"),
        "acc":  accuracy_score(y_true, y_pred),
        "prec": precision_score(y_true, y_pred, average="macro"),
        "rec":  recall_score(y_true, y_pred, average="macro")
    }

if __name__ == "__main__":
    CSV = r"C:\Users\ernes\OneDrive\Escritorio\Reto Final\4. Modelo\data_all.csv"
    texts, y_true = load_data(CSV)
    y_pred, y_proba = llama_classify(texts)
    m = metrics(y_true, y_pred, y_proba)
    print("\n=== Zero-Shot Llama-3-8B-Instruct ===")
    for k,v in m.items(): print(f"{k.upper():5}: {v:.4f}")
