from fastapi.testclient import TestClient

from api import app

client = TestClient(app)


def test_home():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "API is running"


def test_prediction():
    response = client.post(
        "/predict",
        json={
            "sepal_length": 5.1,
            "sepal_width": 3.5,
            "petal_length": 1.4,
            "petal_width": 0.2,
        },
    )

    assert response.status_code == 200

    result = response.json()

    assert "prediction" in result
    assert "confidence" in result
    assert result["prediction"] in [
        "setosa",
        "versicolor",
        "virginica",
    ]
