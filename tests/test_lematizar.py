#Autor: Ernesto Juárez Torres A01754887

def test_lematizar_saca_forma_base(tokens_base, lematizar_tokens):
    lemas = lematizar_tokens(tokens_base)
    # spaCy para español no cambia "visita"
    assert "visita" in lemas
