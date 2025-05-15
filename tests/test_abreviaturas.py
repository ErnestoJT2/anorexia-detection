#Autor: Ernesto Juárez Torres A01754887

def test_expandir_abreviaturas(tokens_base, expandir_abreviaturas):
    exp = expandir_abreviaturas(tokens_base)
    assert "please" in exp
    assert "pls" not in exp
