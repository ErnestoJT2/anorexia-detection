#Autor: Ernesto Juárez Torres A01754887

def test_stylistic_columns(tiny_corpus, stylistic_matrix):
    M = stylistic_matrix(tiny_corpus)
    assert M.shape == (2,3)            
    assert M[0,0] == 4
