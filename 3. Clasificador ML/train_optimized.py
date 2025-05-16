# train_optimized.py
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, roc_auc_score, f1_score, RocCurveDisplay
from scipy.special import expit

# === Cargar datos ===
BASE = Path(__file__).parent
OUT = BASE / "out"
train_df = pd.read_csv(OUT / "train.csv")
val_df = pd.read_csv(OUT / "val.csv")

X_train = train_df.drop(columns=["class"])
y_train = train_df["class"]
X_val = val_df.drop(columns=["class"])
y_val = val_df["class"]

resultados = []

# === Random Forest optimizado ===
gs = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid={"n_estimators": [100, 200], "max_depth": [10, 20, None]},
    scoring="roc_auc", cv=3, n_jobs=-1
)
gs.fit(X_train, y_train)
pred_rf = gs.predict(X_val)
proba_rf = gs.predict_proba(X_val)[:, 1]
auc_rf = roc_auc_score(y_val, proba_rf)
f1_rf = f1_score(y_val, pred_rf)

with open(OUT / "reporte_rf_opt.txt", "w") as f:
    f.write(classification_report(y_val, pred_rf, target_names=["control", "anorexia"]))

RocCurveDisplay.from_predictions(y_val, proba_rf).plot()
plt.title("ROC – RF Optimizado")
plt.tight_layout()
plt.savefig(OUT / "roc_rf_opt.png")
plt.close()

resultados.append({"modelo": "rf_opt", "AUC": auc_rf, "F1": f1_rf})

print("✅ RF optimizado entrenado")
print(f"🔹 AUC: {auc_rf:.3f} | F1: {f1_rf:.3f}")

# === SVM balanceado ===
svm = LinearSVC(class_weight="balanced", max_iter=10000)
svm.fit(X_train, y_train)
scores_svm = expit(svm.decision_function(X_val))
pred_svm = svm.predict(X_val)
auc_svm = roc_auc_score(y_val, scores_svm)
f1_svm = f1_score(y_val, pred_svm)

with open(OUT / "reporte_svm_bal.txt", "w") as f:
    f.write(classification_report(y_val, pred_svm, target_names=["control", "anorexia"]))

RocCurveDisplay.from_predictions(y_val, scores_svm).plot()
plt.title("ROC – SVM Balanceado")
plt.tight_layout()
plt.savefig(OUT / "roc_svm_bal.png")
plt.close()

resultados.append({"modelo": "svm_bal", "AUC": auc_svm, "F1": f1_svm})

print("✅ SVM balanceado entrenado")
print(f"🔹 AUC: {auc_svm:.3f} | F1: {f1_svm:.3f}")

# === Guardar métricas ===
pd.DataFrame(resultados).to_csv(OUT / "metrics.csv", mode="a", index=False, header=not (OUT / "metrics.csv").exists())
