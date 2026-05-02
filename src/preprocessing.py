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
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

def fetch_data(table="diabetes"):
    print("🔗 Conectando ao Supabase...")

    sb = create_client(SUPABASE_URL, SUPABASE_KEY)

    res = sb.table(table).select("*").execute()
    df = pd.DataFrame(res.data)

    df.columns = df.columns.str.lower()

    print(f"📊 Dados carregados: {df.shape}")
    return df



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
            -- colunas originais
            pregnancies,
            age,
            diabetespedigreefunction,
            outcome,

            -- limpeza + imputação
            COALESCE(NULLIF(glucose, 0), MEDIAN(NULLIF(glucose, 0)) OVER()) AS glucose,
            COALESCE(NULLIF(bloodpressure, 0), MEDIAN(NULLIF(bloodpressure, 0)) OVER()) AS bloodpressure,
            COALESCE(NULLIF(skinthickness, 0), MEDIAN(NULLIF(skinthickness, 0)) OVER()) AS skinthickness,
            COALESCE(NULLIF(insulin, 0), MEDIAN(NULLIF(insulin, 0)) OVER()) AS insulin,
            COALESCE(NULLIF(bmi, 0), MEDIAN(NULLIF(bmi, 0)) OVER()) AS bmi,

            -- feature engineering
            CASE
                WHEN glucose > 125 THEN 1
                ELSE 0
            END AS high_glucose_flag,
        FROM df
    """).df()

    return df_sql


# def clean_data(df):
#     print("🧹 Limpando dados...")

#     cols = ['glucose', 'bloodpressure', 'skinthickness', 'insulin', 'bmi']

#     print("\n🚨 Zeros inválidos antes:")
#     for col in cols:
#         print(f"{col}: {(df[col] == 0).sum()}")

#     # substituir zeros por NaN
#     df[cols] = df[cols].replace(0, pd.NA)

#     # imputação com mediana
#     df = df.fillna(df.median())

#     print("\n✅ Dados limpos!")

#     return df


def save_data(df, path="data/processed/processed.parquet"):
    print("💾 Salvando dados processados...")

    os.makedirs("data/processed", exist_ok=True)

    df.to_parquet(path, index=False)

    print(f"Arquivo salvo em: {path}")
    print(f"Shape final: {df.shape}")


def main():
    df = fetch_data()
    df = process_with_duckdb(df)    

    save_data(df)


if __name__ == "__main__":
    main()