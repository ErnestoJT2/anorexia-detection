# 🧠 Anorexia Detection Pipeline

Este repositorio contiene un **pipeline completo** para la detección de señales de trastornos alimenticios (específicamente anorexia) en texto, usando datos similares a publicaciones de Twitter o Reddit. Incluye:

1. **Preprocesamiento** del texto (ruido, tokenización, lematización, abreviaturas)
2. **Extracción de atributos** (TF-IDF, BoW, n-grams, palabras clave, sentimiento, estilo)
3. **Clasificación** supervisada (Random Forest y SVM/Logistic Regression)
4. **API REST** para inferencia en tiempo real con FastAPI

---

## 📦 Clonar este repositorio

```bash
git clone https://github.com/ErnestoJT2/anorexia-detection.git
cd anorexia-detection
```

---

## 📁 Estructura de carpetas

```
.
├── 1. Preprocesamiento de Texto/
│   ├── *.py               # Scripts de limpieza y procesamiento
│   └── data_final.csv     # Texto ya procesado
│
├── 2. Extraccion de Atributos/
│   ├── tfidf.py, bow.py, ngrams.py, ...
│   └── out/               # tfidf.npz, bow.npz, keywords.npy, etc.
│
├── 3. Clasificador ML/
│   ├── split_data.py      # División train/valid/test
│   ├── train_baseline.py
│   ├── train_optimized.py
│   ├── dataload.py
│   ├── splits/            # train_idx.npy, valid_idx.npy, test_idx.npy
│   └── out/               # Modelos, ROC, métricas
│
├── 4. Modelo/
│   └── predict.py         # API con FastAPI
│   └── anorexia.html      # Interfaz de usuario
│
├── tests/                 # Pruebas automáticas
├── requirements.txt
└── README.md
```

---

## 🚀 Crear entorno de desarrollo

Desde la carpeta raíz del proyecto:

```powershell
py -3.10 -m venv .venv310
.venv310\Scripts\Activate.ps1  # En Windows PowerShell
```

o en Mac/Linux:

```bash
python3.10 -m venv .venv310
source .venv310/bin/activate
```

### Instalar dependencias:

```bash
pip install -r requirements.txt
```

### Descargar recursos adicionales:

```bash
python -m nltk.downloader punkt stopwords
python -m spacy download es_core_news_sm
```

---

## ⚙️ Uso paso a paso

### 1. Preprocesar texto (si no existe `data_final.csv`)

```bash
python "1. Preprocesamiento de Texto/pipeline_driver.py"
```

### 2. Extraer atributos

```bash
python "2. Extraccion de Atributos/pipeline_features.py"
```

### 3. Crear splits y entrenar modelos

```bash
python "3. Clasificador ML/split_data.py"
python "3. Clasificador ML/train_baseline.py"
python "3. Clasificador ML/train_optimized.py"
```

### 4. Combinar gráficas ROC

```bash
python "3. Clasificador ML/plots.py"
```

---

## 🧪 Ejecutar pruebas

```bash
pytest -q
```

---

## 🌐 API de inferencia

Iniciar el servidor:

```bash
uvicorn "4. Modelo.predict:app" --reload --host 127.0.0.1 --port 8000
```

Abrir en el navegador:

```
http://127.0.0.1:8000/anorexia.html
```

---

## 📮 Usar API vía POST

### Ejemplo (Postman o curl):

```
POST http://127.0.0.1:8000/predict
Content-Type: application/json

{
  "text": "Me siento sin esperanza y con atracones constantes"
}
```

### Respuesta esperada:

```json
{
  "prediction": "anorexia",
  "probability": 0.87
}
```

---

## 📊 Resultados generados

* `roc_random_forest.png`, `roc_svm.png`, `roc_all.png`
* `metrics.csv`, `metrics_heatmap.png`
* Modelos entrenados: `*.joblib`

---

## 📝 Licencia

Proyecto creado con fines académicos. Puedes reutilizarlo para proyectos educativos mencionando al autor.
