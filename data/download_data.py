"""
download_data.py — Baixa dataset do Kaggle e salva em data/raw/

⚠️ IMPORTANTE:
Este script NÃO instala dependências automaticamente.

Certifique-se de ter instalado previamente via requirements.txt:

    pip install -r requirements.txt

Dependências necessárias:
    - kaggle
    - pandas
"""

import os
import pandas as pd
from kaggle.api.kaggle_api_extended import KaggleApi


def download_dataset():
    print("📥 Baixando dataset do Kaggle...")

    api = KaggleApi()
    api.authenticate()

    dataset = "uciml/pima-indians-diabetes-database"

    os.makedirs("data/raw", exist_ok=True)

    api.dataset_download_files(
        dataset,
        path="data/raw",
        unzip=True
    )

    print("✅ Download concluído!")


def load_and_describe():
    path = "data/raw/diabetes.csv"

    df = pd.read_csv(path)

    print("\n📊 Estatísticas Descritivas:")
    print(df.describe().round(2))

    print("\n🔎 Informações do dataset:")
    print(df.info())

    print("\n🔎 Colunas do dataset:")
    print(df.columns)

    return df


if __name__ == "__main__":
    download_dataset()
    load_and_describe()