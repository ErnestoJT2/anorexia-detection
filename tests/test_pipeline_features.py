#Autor: Ernesto Juárez Torres A01754887

from pathlib import Path
import numpy as np, scipy.sparse as sp


def test_pipeline_runs(tmp_path, monkeypatch, pipeline_features, tiny_corpus):
    """
    • Redirige pipeline_features.OUT a un directorio temporal.
    • Si el módulo define main(), lo llama.
      ─ Si no, genera al menos tfidf.npz y keywords.npy “a mano”.
    • Comprueba que los archivos existen y tienen forma coherente.
    """
    monkeypatch.setattr(pipeline_features, "OUT", tmp_path, raising=False)
    pipeline_features.OUT.mkdir(exist_ok=True)

    if hasattr(pipeline_features, "main"):
        pipeline_features.main()
    else:
        # --- fallback minimal: creamos dos salidas fundamentales -----------
        tfidf = pipeline_features.tfidf_features(tiny_corpus, max_features=20)
        sp.save_npz(tmp_path / "tfidf.npz", tfidf)

        kw = pipeline_features.keywords_matrix(tiny_corpus)
        np.save(tmp_path / "keywords.npy", kw)
        # -------------------------------------------------------------------

    # verificaciones
    assert (tmp_path / "tfidf.npz").exists()
    assert (tmp_path / "keywords.npy").exists()

    tfidf = sp.load_npz(tmp_path / "tfidf.npz")
    kw = np.load(tmp_path / "keywords.npy")
    assert tfidf.shape[0] == kw.shape[0] > 0
