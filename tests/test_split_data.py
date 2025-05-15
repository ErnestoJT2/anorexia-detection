"""
#Autor: Ernesto Juárez Torres A01754887

Verifica que split_data.py genere índices estratificados
y que las proporciones sean ~70/15/15.
"""

import importlib.util
from pathlib import Path
import numpy as np

def _locate_script():
    """Tolera ambos nombres de carpeta."""
    for p in [
        Path("3. Clasificador ML/split_data.py"),
        Path("3. Ejecución de un clasificador de Machine Learning/split_data.py"),
    ]:
        if p.exists():
            return p
    raise FileNotFoundError("No se encontró split_data.py")

def test_split_indices():
    script = _locate_script()
    spec   = importlib.util.spec_from_file_location("split", script)
    mod    = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)          # ejecuta y crea archivos en mod.SPLITS

    splits_dir = mod.SPLITS
    train = np.load(splits_dir / "train_idx.npy")
    valid = np.load(splits_dir / "valid_idx.npy")
    test  = np.load(splits_dir / "test_idx.npy")
    total = len(train) + len(valid) + len(test)

    assert abs(len(train)/total - 0.70) < 0.02
    assert abs(len(valid)/total - 0.15) < 0.02
    assert abs(len(test) /total - 0.15) < 0.02
