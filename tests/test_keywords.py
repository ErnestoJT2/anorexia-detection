#Autor: Ernesto Juárez Torres A01754887

import numpy as np
def test_keywords_freq(tiny_corpus, keywords_matrix):
    M = keywords_matrix(tiny_corpus)
    assert M.shape == (2, 6)           # 6 keywords definidas
    # en primer doc hay 'comida' y 'atracón'
    assert np.array_equal(M[0, :2], np.array([1,1]))
