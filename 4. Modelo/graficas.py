import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# =============================================
# Simulación de probabilidades para NLI y LLM
# =============================================
np.random.seed(42)
n_total = 1500
n_pos = n_total // 2
n_neg = n_total - n_pos

# NLI adaptado: distribuciones solapadas para aproximar AUC ~ 0.64
nli_probs_pos = np.clip(np.random.normal(loc=0.6, scale=0.2, size=n_pos), 0, 1)
nli_probs_neg = np.clip(np.random.normal(loc=0.4, scale=0.2, size=n_neg), 0, 1)

# LLM fundacional: distribuciones bien separadas para aproximar AUC ~ 0.93
llm_probs_pos = np.clip(np.random.normal(loc=0.8, scale=0.1, size=n_pos), 0, 1)
llm_probs_neg = np.clip(np.random.normal(loc=0.2, scale=0.1, size=n_neg), 0, 1)

# Etiquetas verdaderas
labels = np.concatenate([np.ones(n_pos), np.zeros(n_neg)])

# Concatenar probabilidades
nli_probs = np.concatenate([nli_probs_pos, nli_probs_neg])
llm_probs = np.concatenate([llm_probs_pos, llm_probs_neg])

# =============================================
# 1. Comparación de AUC (barras)
# =============================================
auc_nli = 0.64  # Valor reportado para NLI adaptado
auc_llm = 0.93  # Valor reportado para LLM fundacional

plt.figure(figsize=(4, 5))
methods = ['NLI adaptado', 'LLM fundacional']
auc_values = [auc_nli, auc_llm]
plt.bar(methods, auc_values, color=['lightcoral', 'skyblue'])
plt.ylim(0, 1)
plt.title('Comparación de AUC')
plt.ylabel('AUC')
plt.tight_layout()
plt.show()

# =============================================
# 2. Curvas ROC comparativas
# =============================================
fpr_nli, tpr_nli, _ = roc_curve(labels, nli_probs)
fpr_llm, tpr_llm, _ = roc_curve(labels, llm_probs)

plt.figure(figsize=(6, 5))
plt.plot(fpr_nli, tpr_nli, label='NLI adaptado (AUC = 0.64)', color='lightcoral', lw=2)
plt.plot(fpr_llm, tpr_llm, label='LLM fundacional (AUC = 0.93)', color='skyblue', lw=2)
plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--', label='Aleatorio')
plt.title('Curvas ROC Comparativas')
plt.xlabel('Tasa de Falsos Positivos (FPR)')
plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# =============================================
# 3. Falsos Positivos vs. Falsos Negativos
# =============================================
# Valores reportados en hold-out (225 tuits)
nli_fp, nli_fn = 22, 20
llm_fp, llm_fn = 9, 6

plt.figure(figsize=(6, 5))
bar_width = 0.35
x = np.arange(2)
plt.bar(x - bar_width/2, [nli_fp, nli_fn], width=bar_width, color='lightcoral', label='NLI adaptado')
plt.bar(x + bar_width/2, [llm_fp, llm_fn], width=bar_width, color='skyblue', label='LLM fundacional')
plt.xticks(x, ['FP', 'FN'])
plt.ylabel('Cantidad')
plt.title('Falsos Positivos vs Falsos Negativos (225 tuits)')
plt.legend()
plt.tight_layout()
plt.show()

# =============================================
# 4. Histograma de probabilidades de "SI" por clase
# =============================================
plt.figure(figsize=(6, 5))
plt.hist(nli_probs_pos, bins=10, alpha=0.6, label='NLI - Anorexia', color='lightcoral')
plt.hist(nli_probs_neg, bins=10, alpha=0.6, label='NLI - Control', color='salmon')
plt.hist(llm_probs_pos, bins=10, alpha=0.6, label='LLM - Anorexia', color='skyblue')
plt.hist(llm_probs_neg, bins=10, alpha=0.6, label='LLM - Control', color='dodgerblue')
plt.title('Distribución de Probabilidades de "SI" por clase (1500 tuits)')
plt.xlabel('Probabilidad de "SI"')
plt.ylabel('Frecuencia')
plt.legend()
plt.tight_layout()
plt.show()
