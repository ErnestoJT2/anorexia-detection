# run_zero_shot_nli.py

import pandas as pd
import torch 
from transformers import XLMRobertaTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score
)
from tqdm import tqdm

def load_data(path_csv: str):
    df = pd.read_csv(path_csv)
    texts = df["tweet_text"].tolist()
    labels = [1 if c == "anorexia" else 0 for c in df["class"].tolist()]
    return texts, labels

def zero_shot_nli_predict(texts, tokenizer, model, device):
    entail_idx = model.config.label2id["entailment"]
    preds, probs = [], []

    for txt in tqdm(texts, desc="Zero-Shot NLI"):
        encoded = tokenizer(
            txt,
            "Este texto expresa intenciones de quitarse la vida.",
            return_tensors="pt",
            truncation="only_first",
            max_length=512,
            padding="max_length"
        ).to(device)

        with torch.no_grad():
            logits = model(**encoded).logits
            proba_entail = softmax(logits, dim=1).cpu().numpy()[0][entail_idx]

        preds.append(1 if proba_entail > 0.5 else 0)
        probs.append(proba_entail)

    return preds, probs

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")

    model_name = "joeddav/xlm-roberta-large-xnli"
    print(f"Cargando modelo {model_name}...")

    # Usar slow-tokenizer para evitar tiktoken
    tokenizer = XLMRobertaTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

    ruta_all = r"C:\Users\ernes\OneDrive\Escritorio\Reto Final\4. Modelo\data_all.csv"
    texts, labels_true = load_data(ruta_all)
    print(f"Total de ejemplos cargados: {len(texts)}")

    preds, probs = zero_shot_nli_predict(texts, tokenizer, model, device)

    auc = roc_auc_score(labels_true, probs)
    f1 = f1_score(labels_true, preds, average="macro")
    acc = accuracy_score(labels_true, preds)
    prec = precision_score(labels_true, preds, average="macro")
    rec = recall_score(labels_true, preds, average="macro")

    print("\n=== Zero-Shot NLI (XLM-RoBERTA-XNLI) sobre 1500 ejemplos ===")
    print(f"AUC:     {auc:.4f}")
    print(f"F1-macro:{f1:.4f}")
    print(f"Accuracy:{acc:.4f}")
    print(f"Prec-macro:{prec:.4f}")
    print(f"Rec-macro: {rec:.4f}")

if __name__ == "__main__":
    main()
# Save results