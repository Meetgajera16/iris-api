import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier

app = FastAPI(title="Iris Prediction API", version="1.0")

X, y = load_iris(return_X_y=True)

model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X, y)


class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


@app.get("/")
def home():
    return {"status": "API is running"}


@app.post("/predict")
def predict(data: IrisInput):
    features = np.array(
        [[data.sepal_length, data.sepal_width, data.petal_length, data.petal_width]]
    )

    prediction = int(model.predict(features)[0])
    probability = model.predict_proba(features)[0]

    classes = ["setosa", "versicolor", "virginica"]

    return {"prediction": classes[prediction], "confidence": float(max(probability))}
