#Autor: Ernesto Juárez Torres A01754887

import pytest
from fastapi.testclient import TestClient 

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), r'..\4. Modelo')))
from predict import app 

client = TestClient(app)

@pytest.mark.parametrize("input_text,expected_prediction", [
    ("Me siento culpable por comer tanto hoy", "anorexia"),
    ("Tuve un día muy saludable y balanceado", "control"),
    ("Me preocupa mucho mi peso últimamente", "anorexia"),
    ("Estoy contento con mis hábitos alimenticios", "control")
])
def test_prediction(input_text, expected_prediction):
    response = client.post("/predict", json={"text": input_text})
    assert response.status_code == 200
    result = response.json()
    assert "prediction" in result
    assert result["prediction"] == expected_prediction
    assert "probability" in result
    assert 0 <= result["probability"] <= 1
