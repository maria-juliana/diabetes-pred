"""
train.py — Treina o modelo de análise de sentimento e registra no MLflow/DagsHub.

Execute UMA VEZ por experimento, mudando os parâmetros marcados com # MUDE AQUI:
    python src/train.py

Experimento 1: LogisticRegression (C=1)
Experimento 2: LogisticRegression (C=0.1)
"""

import os
import mlflow
import mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score

from dotenv import load_dotenv
from src.prepare_data import load_and_split

# ── Credenciais ─────────────────────────────────────────
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

# ═══════════════════════════════════════
# PARÂMETROS DO MODELO
# ═══════════════════════════════════════
C = 0.1
RUN_NAME = "baseline-logreg2"
# ═══════════════════════════════════════


def main():
    X_train, X_test, y_train, y_test = load_and_split()

    print(f"Treino: {len(X_train)} | Teste: {len(X_test)}")

    with mlflow.start_run(run_name=RUN_NAME):

        #Log de parâmetros
        mlflow.log_params({
            "model": "LogisticRegression",
            "C": C,
            "scaler": "StandardScaler",
            "test_size": 0.2,
            "dataset": "data/diabetes.csv",
        })

        #Pipeline
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                C=C,
                max_iter=1000
            )),
        ])

        #Treino
        pipeline.fit(X_train, y_train)

        #Predição
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        #Métricas
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc = roc_auc_score(y_test, y_proba)

        print("\n" + "=" * 50)
        print(classification_report(y_test, y_pred))
        print("=" * 50)

        #Log de métricas
        mlflow.log_metrics({
            "accuracy": acc,
            "f1": f1,
            "roc_auc": roc,
        })

        #Log do modelo
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            registered_model_name="DiabetesClassifier",
        )

        print(f"\nRun '{RUN_NAME}' finalizado!")
        print(f"Acurácia : {acc:.2%}")
        print(f"F1       : {f1:.3f}")
        print(f"ROC-AUC  : {roc:.3f}")


if __name__ == "__main__":
    main()