"""
prepare_data.py — Carrega, limpa e divide o dataset de reviews.

Uso standalone:
    python src/prepare_data.py

Uso como módulo (importado pelo train.py e evaluate.py):
    from src.prepare_data import load_and_split
"""

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_and_split(
    path: str = str(BASE_DIR / "data" / "diabetes.csv"),
    test_size: float = 0.2,
    seed: int = 42,
):

    df = pd.read_csv(path)

    #Tratamento de zeros inválidos
    features_with_invalid_zeros = [
        'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI'
    ]

    print("\nInvalid Zero Values Count (Before Processing):")
    for col in features_with_invalid_zeros:
        print(f"{col}: {(df[col] == 0).sum()}")

    df[features_with_invalid_zeros] = df[features_with_invalid_zeros].replace(0, np.nan)

    #Preenchimento com mediana
    df.fillna(df.median(), inplace=True)

    #Separação de features e target
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    #Split
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_and_split()

    print(f"\nTotal de amostras de treino : {len(X_train)}")
    print(f"Total de amostras de teste  : {len(X_test)}")

    print("\nDistribuição de classes no treino:")
    print(y_train.value_counts())