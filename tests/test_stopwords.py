#Autor: Ernesto Juárez Torres A01754887

def test_elimina_stopwords(tokens_base, eliminar_stopwords):
    sin_sw = eliminar_stopwords(tokens_base)
    # 'pls' no es stop‑word, 'hola' tampoco. Verifiquemos una real:
    assert "la" not in eliminar_stopwords(["hola", "la", "casa"])
