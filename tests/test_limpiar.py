#Autor: Ernesto Juárez Torres A01754887

def test_limpiar_quita_ruido(texto_raw, limpiar_texto):
    limpio = limpiar_texto(texto_raw)
    assert "https://" not in limpio
    assert "@" not in limpio
    assert "#" not in limpio
    assert "😊" not in limpio
