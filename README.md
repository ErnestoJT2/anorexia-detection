
# 🧠 Clasificador de Anorexia en Tweets – Reto TC3002B

Este repositorio implementa un sistema completo para detectar señales de **trastornos de la conducta alimentaria (TCA)**, específicamente **anorexia**, en publicaciones escritas en español. Se utiliza un pipeline de procesamiento lingüístico y modelos de aprendizaje automático validados según el protocolo del curso TC3002B.

---

## 📁 Estructura del Proyecto

```
├── 1. Preprocesamiento de Texto/
│   ├── pipeline_driver.py  ← Ejecuta los pasos 1 a 6 de limpieza
│   ├── step1_eliminacion_de_ruido.py
│   ├── step2_normalizacion.py
│   ├── step3_tokenizacion.py
│   ├── step4_eliminacion_stopwords.py
│   ├── step5_lematizacion.py
│   └── step6_abreviaturas.py
│
├── 2. Extraccion de Atributos/
│   ├── pipeline_features.py
│   └── OUT/
│       ├── *.csv (TF-IDF, BoW, keywords, etc.)
│       └── *.png (visualizaciones)
│
├── 3. Clasificador ML/
│   ├── split_data.py
│   ├── train_baseline.py
│   ├── train_optimized.py
│   ├── plot_metrics.py
│   └── out/
│       ├── reporte_*.txt
│       ├── comparacion_modelos.csv
│       ├── roc_*.png
│       └── comparacion_heatmap.png
│
├── 4. Servicio API/
│   └── predict.py
│__ 5. tests/
```

---

## ⚙️ Instalación

```bash
git clone https://github.com/TU_USUARIO/Clasificador-Anorexia-TC3002B.git
cd Clasificador-Anorexia-TC3002B
python -m venv .venv
source .venv/bin/activate  # o .venv\Scripts\activate en Windows
pip install -r requirements.txt
```

---

## 🧪 Ejecución del pipeline

### 1. Preprocesamiento de texto

```bash
python "1. Preprocesamiento de Texto/pipeline_driver.py"
```

### 2. Extracción de atributos

```bash
python "2. Extraccion de Atributos/pipeline_features.py"
```

### 3. División de datos y entrenamiento

```bash
python "3. Clasificador ML/split_data.py"
python "3. Clasificador ML/train_baseline.py"
python "3. Clasificador ML/train_optimized.py"
python "3. Clasificador ML/plot_metrics.py"
```

---

## 📊 Resultados

- **Modelo SVM balanceado:**  
  - AUC: 0.939  
  - F1-score: 0.859  
- Reportes de clasificación: `3. Clasificador ML/out/reporte_*.txt`
- Comparación visual:  
  ![comparacion_heatmap.png](3. Clasificador ML/out/comparacion_heatmap.png)

---

## 🚀 Predicción vía API

```bash
 cd '.\4. Modelo\'                                                                   
 uvicorn predict:app --reload --host 0.0.0.0 --port 8000
```

Accede a la interfaz en tu navegador: [http://127.0.0.1:8000/anorexia.html](http://127.0.0.1:8000/anorexia.html)

Puedes enviar POST como:
```json
{
  "text": "no quiero comer, me siento gorda"
}
```

---

## 🧪 Pruebas automáticas

```bash
pytest -q
```

---

## 📄 Licencia

MIT – Ernesto Juárez Torres, 2025
