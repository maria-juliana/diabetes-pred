"""
validate_env.py — Validação de credenciais e pré-requisitos do pipeline DVC.

Uso:
    python -m src.validate_env
"""

import os
from dotenv import load_dotenv


def _required_vars():
    return {
        "Kaggle": ["KAGGLE_USERNAME", "KAGGLE_KEY"],
        "Supabase": ["SUPABASE_URL", "SUPABASE_KEY"],
        "DagsHub": ["DAGSHUB_USER", "DAGSHUB_REPO", "DAGSHUB_TOKEN"],
    }


def validate_env_vars():
    load_dotenv()

    missing = []

    for group, variables in _required_vars().items():
        for var in variables:
            if not os.getenv(var):
                missing.append((group, var))

    if missing:
        lines = ["Variáveis de ambiente ausentes no .env:"]
        for group, var in missing:
            lines.append(f"- [{group}] {var}")
        raise RuntimeError("\n".join(lines))


def validate_paths():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)


def main():
    print("🔍 Validando ambiente do pipeline...")
    validate_env_vars()
    validate_paths()
    print("✅ Validação concluída com sucesso; nenhum erro foi encontrado.")


if __name__ == "__main__":
    main()
