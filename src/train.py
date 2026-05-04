"""
train.py — Treinamento + MLflow (DagsHub)

Uso:
    python -m src.train
"""

import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
import joblib
from dotenv import load_dotenv


# ── Credenciais ────────────────────────────────────────
load_dotenv()

DAGSHUB_USER  = os.getenv("DAGSHUB_USER")
DAGSHUB_REPO  = os.getenv("DAGSHUB_REPO")
DAGSHUB_TOKEN = os.getenv("DAGSHUB_TOKEN")

mlflow.set_tracking_uri(
    f"https://dagshub.com/{DAGSHUB_USER}/{DAGSHUB_REPO}.mlflow"
)

os.environ["MLFLOW_TRACKING_USERNAME"] = DAGSHUB_USER
os.environ["MLFLOW_TRACKING_PASSWORD"] = DAGSHUB_TOKEN

mlflow.set_experiment("diabetes-classification")


# ═══════════════════════════════════════════════════════
# MUDE PARA EXPERIMENTOS
# ═══════════════════════════════════════════════════════
C = 1.0
RUN_NAME = "exp-1-baseline"
CLASS_WEIGHT = None

#C = 0.1
#RUN_NAME = "exp-2-baseline"
#CLASS_WEIGHT = "balanced"

#C = 10.0
#RUN_NAME = "exp-3-baseline"
#CLASS_WEIGHT = None
# ═══════════════════════════════════════════════════════


def main():
    print("📥 Carregando dados processados...")

    df = pd.read_parquet("data/processed/processed.parquet")

    X = df.drop("outcome", axis=1)
    y = df["outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    with mlflow.start_run(run_name=RUN_NAME):

        # parâmetros
        mlflow.log_params({
            "model": "LogisticRegression",
            "C": C,
            "class_weight": CLASS_WEIGHT
        })

        # modelo
        model = LogisticRegression(max_iter=1000, C=C, class_weight=CLASS_WEIGHT)
        model.fit(X_train, y_train)

        # avaliação
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1  = f1_score(y_test, y_pred)

        print(f"Acurácia: {acc:.2%}")
        print(f"F1-score: {f1:.3f}")

        mlflow.log_metrics({
            "accuracy": acc,
            "f1": f1
        })

        # registrar modelo
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="DiabetesClassifier"
        )

        # salvar local
        joblib.dump(model, "model.pkl")

        print("✅ Modelo treinado!")


if __name__ == "__main__":
    main()
