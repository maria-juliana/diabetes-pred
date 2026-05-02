"""
preprocessing.py — Processamento e feature engineering com DuckDB

Fluxo:
data/raw/raw.csv → DuckDB → limpeza → data/processed/processed.parquet

Uso:
    python -m src.preprocessing
"""

import duckdb
import pandas as pd
import os


def load_data(path="data/raw/raw.csv"):
    print("📥 Carregando dados...")
    df = pd.read_csv(path)
    print(f"Shape inicial: {df.shape}")
    return df


def process_with_duckdb(df):
    print("🦆 Processando com DuckDB...")

    con = duckdb.connect()
    con.register("df", df)

    # ── Exemplo de transformação em SQL ──────────────────
    df_sql = con.execute("""
        SELECT
            *,
            CASE
                WHEN glucose > 125 THEN 1
                ELSE 0
            END AS high_glucose_flag
        FROM df
    """).df()

    return df_sql


def clean_data(df):
    print("🧹 Limpando dados...")

    cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

    print("\n🚨 Zeros inválidos antes:")
    for col in cols:
        print(f"{col}: {(df[col] == 0).sum()}")

    # substituir zeros por NaN
    df[cols] = df[cols].replace(0, pd.NA)

    # imputação com mediana
    df = df.fillna(df.median())

    print("\n✅ Dados limpos!")

    return df


def save_data(df, path="data/processed/processed.parquet"):
    print("💾 Salvando dados processados...")

    os.makedirs("data/processed", exist_ok=True)

    df.to_parquet(path, index=False)

    print(f"Arquivo salvo em: {path}")
    print(f"Shape final: {df.shape}")


def main():
    df = load_data()

    df = process_with_duckdb(df)

    df = clean_data(df)

    save_data(df)


if __name__ == "__main__":
    main()