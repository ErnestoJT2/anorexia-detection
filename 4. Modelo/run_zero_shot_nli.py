# run_zero_shot_nli_con_graficas.py

import re
import pandas as pd
import torch
from transformers import XLMRobertaTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    accuracy_score,
    precision_score,
    recall_score,
    roc_curve,
    precision_recall_curve
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# ------------------------------------------------------------
# 1. Función de limpieza de texto
# ------------------------------------------------------------
def limpiar_texto(texto: str) -> str:
    """
    - Pasa a minúsculas.
    - Elimina URLs, menciones (@usuario) y hashtags (#etiqueta).
    - Elimina caracteres de puntuación salvo letras acentuadas y espacios.
    - Colapsa espacios múltiples.
    """
    texto = texto.lower()
    texto = re.sub(r"http\S+|www\.\S+", "", texto)
    texto = re.sub(r"@[A-Za-z0-9_]+", "", texto)
    texto = re.sub(r"#[A-Za-z0-9_]+", "", texto)
    texto = re.sub(r"[^\wáéíóúüñ\s]", "", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto

# ------------------------------------------------------------
# 2. Carga de datos
# ------------------------------------------------------------
def load_data(path_csv: str):
    """
    Lee el CSV que contiene 'tweet_text' y 'class'.
    Convierte 'class' == "anorexia" en etiqueta 1, cualquier otro valor en 0.
    Retorna: lista de tuits (texts) y lista de etiquetas (labels).
    """
    df = pd.read_csv(path_csv)
    texts = df["tweet_text"].tolist()
    labels = [1 if c == "anorexia" else 0 for c in df["class"].tolist()]
    return texts, labels

# ------------------------------------------------------------
# 3. Inferencia zero-shot NLI en lotes (con las 3 probabilidades)
# ------------------------------------------------------------
def zero_shot_nli_predict(texts, tokenizer, model, device, batch_size=16, threshold=0.50):
    """
    - texts: lista de tuits sin procesar.
    - tokenizer, model: XLM-RoBERTa NLI preentrenado.
    - device: "cuda" o "cpu".
    - batch_size: tamaño del lote para tokenizar en bloque.
    - threshold: umbral para asignar etiqueta 1 si p_entail >= threshold.
    Retorna:
      preds           : lista de etiquetas finales (0 o 1).
      probs_entail    : lista de probabilidades de 'entailment'.
      probs_neutral   : lista de probabilidades de 'neutral'.
      probs_contradict: lista de probabilidades de 'contradiction'.
    """
    # Obtenemos los índices de cada etiqueta en la salida logits
    idx_contra = model.config.label2id["contradiction"]
    idx_neutral = model.config.label2id["neutral"]
    idx_entail = model.config.label2id["entailment"]

    preds = []
    probs_entail = []
    probs_neutral = []
    probs_contradict = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Zero-Shot NLI (batches)"):
        batch_texts = texts[i : i + batch_size]
        # 3.1. Limpiar cada tuit del lote
        batch_texts = [limpiar_texto(t) for t in batch_texts]

        # 3.2. Hipótesis repetida (misma para cada tuit)
        hipotesis = ["Este texto indica conductas asociadas a la anorexia."] * len(batch_texts)

        # 3.3. Tokenizar el lote completo
        encoded = tokenizer(
            batch_texts,
            hipotesis,
            return_tensors="pt",
            truncation="only_first",
            max_length=512,
            padding="max_length"
        ).to(device)

        # 3.4. Inferencia sin gradientes
        with torch.no_grad():
            logits = model(**encoded).logits               # shape: [batch_size, 3]
            probs_batch = softmax(logits, dim=1).cpu().numpy()
            # probs_batch[i] = [p_contradiction, p_neutral, p_entailment] para cada ejemplo i

        # 3.5. Extraer cada probabilidad y asignar etiqueta según threshold
        for p_vec in probs_batch:
            p_c = p_vec[idx_contra]
            p_n = p_vec[idx_neutral]
            p_e = p_vec[idx_entail]

            probs_contradict.append(p_c)
            probs_neutral.append(p_n)
            probs_entail.append(p_e)

            # Predicción binaria basada en p_entail
            preds.append(1 if p_e >= threshold else 0)

    return preds, probs_entail, probs_neutral, probs_contradict

# ------------------------------------------------------------
# 4. Función principal
# ------------------------------------------------------------
def main():
    # 4.1. Selección de dispositivo
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Usando dispositivo: {device}")

    # 4.2. Carga del modelo preentrenado XLM-RoBERTa NLI
    model_name = "joeddav/xlm-roberta-large-xnli"
    print(f"Cargando modelo {model_name}...")
    tokenizer = XLMRobertaTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)

    # 4.3. Lectura de datos
    ruta_all = r"C:\Users\ernes\OneDrive\Escritorio\Reto Final\4. Modelo\data_all.csv"
    texts, labels_true = load_data(ruta_all)
    print(f"Total de ejemplos cargados: {len(texts)}")

    # 4.4. Separar un 10% para validación (opcional, para calibrar umbral)
    texts_train, texts_val, labels_train, labels_val = train_test_split(
        texts, labels_true, test_size=0.10, random_state=42, stratify=labels_true
    )

    # 4.5. Obtener probabilidades en validación (sin threshold) para calibrar umbral
    _, probs_ent_val, probs_neu_val, probs_con_val = zero_shot_nli_predict(
        texts_val, tokenizer, model, device, batch_size=16, threshold=0.0
    )

    # 4.6. Encontrar el mejor umbral en [0.30, 0.70] para maximizar F1-macro
    import numpy as np
    mejor_f1 = 0.0
    mejor_umbral = 0.50
    for thr in np.arange(0.30, 0.71, 0.05):
        preds_val = [1 if p >= thr else 0 for p in probs_ent_val]
        f1_thr = f1_score(labels_val, preds_val, average="macro")
        if f1_thr > mejor_f1:
            mejor_f1 = f1_thr
            mejor_umbral = thr

    print(f"\nMejor umbral encontrado en validación: {mejor_umbral:.2f} → F1-macro: {mejor_f1:.4f}")

    # 4.7. Con ese umbral, predecir sobre TODO el conjunto
    preds, probs_entail, probs_neutral, probs_contradict = zero_shot_nli_predict(
        texts, tokenizer, model, device, batch_size=16, threshold=mejor_umbral
    )

    # 4.8. Cálculo de métricas finales (solo con p_entailment)
    auc = roc_auc_score(labels_true, probs_entail)
    f1 = f1_score(labels_true, preds, average="macro")
    acc = accuracy_score(labels_true, preds)
    prec = precision_score(labels_true, preds, average="macro")
    rec = recall_score(labels_true, preds, average="macro")

    print("\n=== Zero-Shot NLI (XLM-RoBERTA-XNLI) sobre 1,500 ejemplos ===")
    print(f"Umbral usado:   {mejor_umbral:.2f}")
    print(f"AUC:            {auc:.4f}")
    print(f"F1-macro:       {f1:.4f}")
    print(f"Accuracy:       {acc:.4f}")
    print(f"Prec-macro:     {prec:.4f}")
    print(f"Rec-macro:      {rec:.4f}")

    # ------------------------------------------------------------
    # 5. GRÁFICAS CON MATPLOTLIB
    # ------------------------------------------------------------
    import matplotlib.pyplot as plt

    # 5.1. Curva ROC
    fpr, tpr, thresholds_roc = roc_curve(labels_true, probs_entail)
    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, linewidth=2)
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1, color="gray")
    plt.title("Curva ROC")
    plt.xlabel("False Positive Rate (FPR)")
    plt.ylabel("True Positive Rate (TPR)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 5.2. Curva Precision-Recall
    precision_vals, recall_vals, thresholds_pr = precision_recall_curve(labels_true, probs_entail)
    plt.figure(figsize=(6, 4))
    plt.plot(recall_vals, precision_vals, linewidth=2)
    plt.title("Curva Precision-Recall")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 5.3. Gráfico comparativo de FP vs. FN
    # ------------------------------------------------------------
    # Calculamos cuatro vectores booleanos:
    #   TP: pred=1 & true=1
    #   TN: pred=0 & true=0
    #   FP: pred=1 & true=0
    #   FN: pred=0 & true=1
    # Luego contamos cuántos FP y cuántos FN hay en todo el conjunto.
    labels_arr = np.array(labels_true)
    preds_arr = np.array(preds)

    fp_mask = (preds_arr == 1) & (labels_arr == 0)
    fn_mask = (preds_arr == 0) & (labels_arr == 1)

    count_fp = np.sum(fp_mask)
    count_fn = np.sum(fn_mask)

    plt.figure(figsize=(6, 4))
    categorias = ["False Positives", "False Negatives"]
    valores = [count_fp, count_fn]
    x_pos = np.arange(len(categorias))
    plt.bar(x_pos, valores, color=["#e74c3c", "#3498db"])
    plt.xticks(x_pos, categorias, rotation=0)
    plt.ylabel("Cantidad de ejemplos")
    plt.title("Comparativo: False Positives vs. False Negatives")
    plt.tight_layout()
    plt.show()

    # 5.4. Histograma de probabilidades de entailment por clase
    # ------------------------------------------------------------
    # Dividimos las probabilidades p_entail según la etiqueta real:
    probs_entail_arr = np.array(probs_entail)

    probs_ent_positivos = probs_entail_arr[labels_arr == 1]
    probs_ent_negativos = probs_entail_arr[labels_arr == 0]

    plt.figure(figsize=(6, 4))
    bins = np.linspace(0, 1, 30)
    plt.hist(probs_ent_negativos, bins=bins, alpha=0.6, label="Clase 0 (control)")
    plt.hist(probs_ent_positivos, bins=bins, alpha=0.6, label="Clase 1 (anorexia)")
    plt.title("Histograma de p_entailment por clase")
    plt.xlabel("p_entailment")
    plt.ylabel("Frecuencia")
    plt.legend()
    plt.tight_layout()
    plt.show()

    # ------------------------------------------------------------
    # 6. Ejemplo de uso de p_contradict y p_neutral
    # ------------------------------------------------------------
    print("\nPrimeros 5 valores de cada probabilidad (para inspección):")
    for i in range(5):
        print(f"Texto {i+1}:")
        print(f"  p_contradiction = {probs_contradict[i]:.4f}")
        print(f"  p_neutral      = {probs_neutral[i]:.4f}")
        print(f"  p_entailment   = {probs_entail[i]:.4f}")
        print(f"  Predicción bin: {preds[i]}")
        print("")

    # Si quieres guardar estas tres columnas en un CSV junto con el texto original y la etiqueta real:
    # df_result = pd.DataFrame({
    #     "tweet_text": texts,
    #     "label_true": labels_true,
    #     "p_contradiction": probs_contradict,
    #     "p_neutral": probs_neutral,
    #     "p_entailment": probs_entail,
    #     "preds_final": preds
    # })
    # df_result.to_csv("resultados_con_probabilidades.csv", index=False)

# ------------------------------------------------------------
# 7. Entry point
# ------------------------------------------------------------
if __name__ == "__main__":
    main()
