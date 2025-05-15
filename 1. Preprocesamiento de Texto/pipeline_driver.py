"""
pipeline_driver.py
Autor: Ernesto Juárez Torres A01754887

Lee `data_train_in.csv` con la columna 'tweet_text', aplica el pipeline
de pre‑procesamiento (ruido → minúsculas → tokenizar → abreviaturas
→ stop‑words → lematizar) y escribe `data_final.csv` con la columna
ya procesada (lista de lemas por tweet).
"""

import importlib.util
from pathlib import Path
import pandas as pd
from tqdm import tqdm  # pip install tqdm

# ---------------- CONFIGURACIÓN -----------------
BASE_DIR = Path(__file__).parent
RUTA_CSV_ENTRADA = BASE_DIR / "data_train_in.csv"
RUTA_CSV_SALIDA = BASE_DIR / "data_final.csv"
COLUMNA_TEXTO = "tweet_text"

# ----------- Función para cargar módulos ----------
_PREPROC_DIR = Path(__file__).parent


def _load(alias: str, filename: str):
    """Carga un módulo con nombre de archivo que no es identificador Python."""
    filepath = _PREPROC_DIR / filename
    spec = importlib.util.spec_from_file_location(alias, str(filepath))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ----------- Cargar cada paso del preprocesamiento ----------
_limpiar = _load("step1_eliminacion_de_ruido", "step1_eliminacion_de_ruido.py")
_normal = _load("step2_normalizacion", "step2_normalizacion.py")
_token = _load("step3_tokenizacion", "step3_tokenizacion.py")
_stopw = _load("step4_eliminacion_stopwords", "step4_eliminacion_stopwords.py")
_lemat = _load("step5_lemantizacion", "step5_lemantizacion.py")
_abrev = _load("step6_abreviaturas", "step6_abreviaturas.py")

limpiar_texto = _limpiar.limpiar_texto
a_minusculas = _normal.a_minusculas
tokenizar_texto = _token.tokenizar_texto
eliminar_stopwords = _stopw.eliminar_stopwords
lematizar_tokens = _lemat.lematizar_tokens
expandir_abreviaturas = _abrev.expandir_abreviaturas
# --------------------------------------------------------------


def procesar(texto: str):
    """
    Encadena los seis pasos de pre‑procesamiento
    y devuelve la lista final de lemas.
    """
    if not isinstance(texto, str) or not texto.strip():
        return []  # Devuelve lista vacía si el texto es nulo o vacío

    texto = limpiar_texto(texto)
    texto = a_minusculas(texto)
    tokens = tokenizar_texto(texto)
    tokens = expandir_abreviaturas(tokens)
    tokens = eliminar_stopwords(tokens)
    lemas = lematizar_tokens(tokens)

    return lemas


def main():
    print(f"📥 Leyendo {RUTA_CSV_ENTRADA}…")
    df = pd.read_csv(RUTA_CSV_ENTRADA)

    # Evitar problemas con NaN/None
    df[COLUMNA_TEXTO] = df[COLUMNA_TEXTO].astype(str)

    print("🚀 Procesando tweets…")
    tqdm.pandas()  # barra de progreso
    df[COLUMNA_TEXTO] = df[COLUMNA_TEXTO].progress_apply(lambda x: procesar(x) if isinstance(x, str) else [])

    print(f"💾 Guardando resultado en {RUTA_CSV_SALIDA}…")
    df.to_csv(RUTA_CSV_SALIDA, index=False)
    print("✅ Preprocesamiento completo.")


if __name__ == "__main__":
    main()
