#Autor: Ernesto Juárez Torres A01754887

from pathlib import Path
import importlib.util
import pandas as pd


def test_pipeline_preproc_end_to_end():
    driver_path = Path("1. Preprocesamiento de Texto") / "pipeline_driver.py"
    assert driver_path.exists(), "pipeline_driver.py no encontrado"

    spec = importlib.util.spec_from_file_location("driver", driver_path)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)

    df = pd.DataFrame({"tweet_text": ["Hola @maria pls 😊 #Prueba"]})
    lemas = df["tweet_text"].map(driver.procesar).iloc[0]

    # spaCy 3.7 ⇒ 'plear', spaCy 3.8 ⇒ 'please'
    assert ("plear" in lemas or "please" in lemas) and "prueba" in lemas
