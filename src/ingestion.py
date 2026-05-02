"""
ingestion.py — Ingestão de dados do Supabase

Fluxo:
Supabase → DataFrame → data/raw/raw.csv

Uso:
    python -m src.ingestion
"""

import os
import pandas as pd
from dotenv import load_dotenv
from supabase import create_client


# ── Carregar variáveis de ambiente ─────────────────────
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

print(SUPABASE_KEY, SUPABASE_URL)

def fetch_data(table_name="diabetes"):
    """
    Conecta ao Supabase e retorna os dados como DataFrame
    """

    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("Credenciais do Supabase não encontradas no .env")

    print("🔗 Conectando ao Supabase...")

    response = supabase.table(table_name).select("*").execute()

    if not response.data:
        raise ValueError("Nenhum dado retornado do Supabase")

    df = pd.DataFrame(response.data)    

    print(f"📊 Dados carregados: {df.shape}")

    return df


def save_raw(df, path="data/raw/raw.csv"):
    """
    Salva os dados localmente
    """
    os.makedirs("data/raw", exist_ok=True)

    df.to_csv(path, index=False)

    print(f"💾 Dados salvos em: {path}")

def upload_in_batches(supabase_client, registros, table_name="diabetes", batch_size=50):
    """
    Insere dados no Supabase em lotes para evitar timeout.

    Parâmetros:
        supabase_client : cliente do Supabase
        registros       : lista de dicionários (df.to_dict('records'))
        table_name      : nome da tabela no Supabase
        batch_size      : tamanho do lote (default=50)

    Retorna:
        total_inserido
    """

    total_inserido = 0
    print(registros)
    print(f"\n📤 Iniciando upload em lotes para tabela '{table_name}'...")

    for i in range(0, len(registros), batch_size):
        lote = registros[i : i + batch_size]

        try:
            supabase_client.table(table_name).insert(lote).execute()

            total_inserido += len(lote)

            print(
                f"  Lote {i // batch_size + 1:02d}: {len(lote)} registros inseridos ✓"
            )

        except Exception as e:
            print(
                f"  Lote {i // batch_size + 1:02d}: erro → {e}"
            )

    print(f"\n✅ Total inserido: {total_inserido} de {len(registros)} registros")

    return total_inserido

if __name__ == "__main__":
    print("🚀 Iniciando ingestão...")

    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)    
    df = pd.read_csv('data/raw/diabetes.csv')
    df.columns = df.columns.str.lower()
    #df = fetch_data()
    registros = df.to_dict(orient='records')
    upload_in_batches(supabase, registros, 'diabetes')

    print("\n🔍 Preview:")
    print(df.head())

    save_raw(df)

    print("\n✅ Ingestão finalizada com sucesso!")