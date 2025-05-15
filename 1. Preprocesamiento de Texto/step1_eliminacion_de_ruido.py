"""
Autor: Ernesto Juárez Torres A01754887

1_eliminacion_de_ruido
======================

Limpia texto crudo proveniente de Twitter/Reddit eliminando URLs,
menciones, emojis, puntuación y caracteres fuera del alfabeto español.

Functions
---------
limpiar_texto(texto: str) -> str
"""

import re
import string

__all__ = ["limpiar_texto"]

# patrón de emojis (mismo que usabas)
_EMOJI_RE = re.compile(
    "["                     # rangos Unicode
    u"\U0001F600-\U0001F64F"
    u"\U0001F300-\U0001F5FF"
    u"\U0001F680-\U0001F6FF"
    u"\U0001F1E0-\U0001F1FF"
    u"\U00002700-\U000027BF"
    u"\U0001F900-\U0001F9FF"
    "]+",
    flags=re.UNICODE,
)

def limpiar_texto(texto: str) -> str:
    """Elimina URLs, menciones, emojis, signos de puntuación y otros caracteres raros."""
    if texto is None:
        return texto

    txt = re.sub(r"https?://\S+|www\.\S+", "", texto)        # URLs
    txt = re.sub(r"@\w+", "", txt)                            # menciones @usuario
    txt = _EMOJI_RE.sub("", txt)                              # emojis
    txt = txt.translate(str.maketrans("", "", string.punctuation))  # puntuación/#hashtags
    txt = re.sub(r"[^0-9A-Za-záéíóúÁÉÍÓÚñÑ\s]", "", txt)      # especiales
    txt = re.sub(r"\s+", " ", txt).strip()                    # espacios de más
    return txt

if __name__ == "__main__":
    print(limpiar_texto("Hola @juan 😊 visita https://x.com #prueba"))
