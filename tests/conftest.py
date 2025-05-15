"""
Autor: Ernesto Juarez Torres A01754887
Fecha: 2025-05

tests/conftest.py
Unifica fixtures para PRE‑PROCESAMIENTO (parte 1) y FEATURES (parte 2).
Funciona con los nombres originales que los tests ya esperan.
"""

import importlib.util
from pathlib import Path
import sys
import pytest

ROOT = Path(__file__).resolve().parent.parent   # carpeta reto-final

# ------------------------------------------------------------------
# helper genérico
# ------------------------------------------------------------------
def _load(module_dir: Path, alias: str, filename: str):
    path = module_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {path}")
    spec = importlib.util.spec_from_file_location(alias, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# ═══════════════════════════ 1 · PREPROCESAMIENTO ═══════════════════════════
PRE_DIR = ROOT / "1. Preprocesamiento de Texto"
sys.path.append(str(PRE_DIR))

step1 = _load(PRE_DIR, "step1", "step1_eliminacion_de_ruido.py")
step2 = _load(PRE_DIR, "step2", "step2_normalizacion.py")
step3 = _load(PRE_DIR, "step3", "step3_tokenizacion.py")
step4 = _load(PRE_DIR, "step4", "step4_eliminacion_stopwords.py")
step5 = _load(PRE_DIR, "step5", "step5_lemantizacion.py")
step6 = _load(PRE_DIR, "step6", "step6_abreviaturas.py")

@pytest.fixture(scope="session")
def limpiar_texto():          return step1.limpiar_texto
@pytest.fixture(scope="session")
def a_minusculas():           return step2.a_minusculas
@pytest.fixture(scope="session")
def tokenizar_texto():        return step3.tokenizar_texto
@pytest.fixture(scope="session")
def eliminar_stopwords():     return step4.eliminar_stopwords
@pytest.fixture(scope="session")
def lematizar_tokens():       return step5.lematizar_tokens
@pytest.fixture(scope="session")
def expandir_abreviaturas():  return step6.expandir_abreviaturas

# fixtures de datos de ejemplo
@pytest.fixture(scope="session")
def texto_raw():
    return "Hola @user! Visita https://x.com 😊 #Prueba pls"

@pytest.fixture(scope="session")
def tokens_base(texto_raw, limpiar_texto, a_minusculas, tokenizar_texto):
    return tokenizar_texto(a_minusculas(limpiar_texto(texto_raw)))

# ═══════════════════════════ 2 · EXTRACCIÓN ATRIBUTOS ═══════════════════════
ATT_DIR = ROOT / "2. Extraccion de Atributos"
sys.path.append(str(ATT_DIR))

m1 = _load(ATT_DIR, "m1", "tfidf.py")
m2 = _load(ATT_DIR, "m2", "bow.py")
m3 = _load(ATT_DIR, "m3", "ngrams.py")
m4 = _load(ATT_DIR, "m4", "salud_mental.py")
m5 = _load(ATT_DIR, "m5", "sentimiento.py")
m6 = _load(ATT_DIR, "m6", "estilisticas.py")
pipe = _load(ATT_DIR, "pipe", "pipeline_features.py")

PRE_DIR = ROOT / "1. Preprocesamiento de Texto"
ATT_DIR = ROOT / "2. Extraccion de Atributos"
ML_DIR  = ROOT / "3. Clasificador ML"

for p in (PRE_DIR, ATT_DIR, ML_DIR):
    sys.path.append(str(p))

# ---- fixtures con nombres EXACTOS que usan los tests ----------------------
@pytest.fixture(scope="session")
def tfidf_features():      return m1.tfidf_features
@pytest.fixture(scope="session")
def bow_features():        return m2.bow_features
@pytest.fixture(scope="session")
def ngram_features():      return m3.ngram_features
@pytest.fixture(scope="session")
def keywords_matrix():     return m4.keywords_matrix
@pytest.fixture(scope="session")
def sentiment_vector():    return m5.sentiment_vector
@pytest.fixture(scope="session")
def stylistic_matrix():    return m6.stylistic_matrix
@pytest.fixture(scope="session")
def pipeline_features():   return pipe
@pytest.fixture(scope="session")
def tiny_corpus():
    return ["hola mundo comida atracón", "anorexia sin esperanza caminando"]