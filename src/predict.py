"""
predict.py — Função de inferência do modelo de diabetes

Uso:
    from src.predict import predict

    predict({
        "pregnancies": 2,
        "glucose": 120,
        "bloodpressure": 70,
        "skinthickness": 20,
        "insulin": 85,
        "bmi": 28.5,
        "diabetespedigreefunction": 0.5,
        "age": 33,
        "high_glucose_flag": 0
    })

RANGES = {
    "pregnancies": (0, 20),
    "glucose": (0, 200),
    "bloodpressure": (0, 140),
    "skinthickness": (0, 100),
    "insulin": (0, 900),
    "bmi": (0.0, 70.0),
    "diabetespedigreefunction": (0.0, 3.0),
    "age": (0, 100)
}
"""

import joblib
import pandas as pd
from sqlalchemy import label
import mlflow
import os
from dotenv import load_dotenv


# ── Carregar modelo ────────────────────────────────────
load_dotenv()

DAGSHUB_USER  = os.getenv("DAGSHUB_USER")
DAGSHUB_REPO  = os.getenv("DAGSHUB_REPO")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")

mlflow.set_tracking_uri(
    f"https://dagshub.com/{DAGSHUB_USER}/{DAGSHUB_REPO}.mlflow"
)

os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USER
os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN


def load_model():
    model = mlflow.sklearn.load_model(
        "models:/DiabetesClassifier@production"
    )
    return model

# ── Função principal ───────────────────────────────────
def predict(data: dict):
    """
    Recebe um dicionário com os dados do paciente
    e retorna a previsão (0 ou 1)
    """
    print(data)
    model = load_model()

    # transforma em DataFrame
    df = pd.DataFrame([data])

    # garante mesmas colunas do treino
    # df.columns = df.columns.str.lower()

    pred = model.predict(df)[0]
    prob = model.predict_proba(df)[0][1]

    label = "Sim" if pred == 1 else "Não"

    return {
    "prediction": label,
    "probability": float(prob)
}


# ── Teste local ────────────────────────────────────────
if __name__ == "__main__":
    sample = {
        "pregnancies": 2,
        "glucose": 130,
        "bloodpressure": 80,
        "skinthickness": 25,
        "insulin": 100,
        "bmi": 30.5,
        "diabetespedigreefunction": 0.6,
        "age": 40,
        "high_glucose_flag": 1
    }

    result = predict(sample)

    print("🔮 Resultado:")
    print(result)