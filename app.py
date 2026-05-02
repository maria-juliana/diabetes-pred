import streamlit as st
import pandas as pd
from src.predict import predict
from dotenv import load_dotenv
import os

load_dotenv()

# ── Config ─────────────────────────────
st.set_page_config(
    page_title="Predição de Diabetes",
    page_icon="🩺",
    layout="wide"
)

st.title("🩺 Predição de Diabetes com Machine Learning")

st.markdown("""
Esta aplicação utiliza um modelo treinado com dados clínicos para estimar o risco de diabetes.

📌 Preencha os dados na aba **Predição** ou explore os dados em **Análise**.
""")

# ── Tabs ─────────────────────────────
tab1, tab2, tab3 = st.tabs(["🔮 Predição", "📊 Análise", "ℹ️ Sobre"])

# ======================================================
# 🔮 TAB 1 — PREDIÇÃO
# ======================================================
with tab1:

    st.subheader("📋 Dados do paciente")

    col1, col2 = st.columns(2)

    with col1:
        pregnancies = st.slider("Gestações", 0, 20, 1)
        glucose = st.slider("Glicose", 0, 200, 100)
        bloodpressure = st.slider("Pressão arterial", 0, 140, 70)
        skinthickness = st.slider("Espessura da pele", 0, 100, 20)

    with col2:
        insulin = st.slider("Insulina", 0, 900, 80)
        bmi = st.slider("BMI", 0.0, 70.0, 25.0)
        dpf = st.slider("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
        age = st.slider("Idade", 0, 100, 30)

    high_glucose_flag = 1 if glucose > 125 else 0

    st.divider()

    if st.button("Qual o risco de diabetes?", use_container_width=True):

        data = {
            "pregnancies": pregnancies,
            "glucose": glucose,
            "bloodpressure": bloodpressure,
            "skinthickness": skinthickness,
            "insulin": insulin,
            "bmi": bmi,
            "diabetespedigreefunction": dpf,
            "age": age,
            "high_glucose_flag": high_glucose_flag
        }

        result = predict(data)

        pred = result["prediction"]
        prob = result["probability"]

        st.subheader("Resultado")

        if pred == "Sim":
            st.error(f"⚠️ Alta probabilidade de diabetes: {prob:.1%}")
        else:
            st.success(f"✅ Baixa probabilidade de diabetes: {prob:.1%}")

        st.markdown("### 📈 Classificação de risco")

        if prob > 0.7:
            st.error("🔴 Alto risco")
        elif prob > 0.4:
            st.warning("🟠 Risco moderado")
        else:
            st.success("🟢 Baixo risco")


# ======================================================
# 📊 TAB 2 — ANÁLISE EXPLORATÓRIA
# ======================================================
with tab2:

    st.subheader("📊 Análise Exploratória")

    df = None

    user = os.getenv("DAGSHUB_USER")
    repo = os.getenv("DAGSHUB_REPO")
    token = os.getenv("DAGSHUB_TOKEN")

    # local
    try:
        df = pd.read_parquet("data/processed/processed.parquet")

    # fallback DagsHub
    except Exception:
        try:
            url = f"https://dagshub.com/{user}/{repo}/raw/main/data/processed/processed.parquet"

            df = pd.read_parquet(
                url,
                storage_options={
                    "Authorization": f"Bearer {token}"
                }
            )

            st.info("📡 Dados carregados do DagsHub")

        except Exception:
            st.warning("⚠️ Dados não disponíveis.")

    if df is not None:
        st.dataframe(df.describe().round(2))
        
# ======================================================
# ℹ️ TAB 3 — SOBRE
# ======================================================
with tab3:

    st.subheader("ℹ️ Sobre o projeto")

    st.markdown("""
### 🧠 Modelo
- Regressão Logística
- Treinado com dados clínicos
- Registrado no MLflow (DagsHub)

### 📊 Features utilizadas
- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

### 📈 Métricas (exemplo)
- Accuracy: ~75–80%
- F1-score: ~0.70+

### ⚙️ Pipeline
- Ingestão: Supabase
- Processamento: DuckDB
- Versionamento: DVC
- Tracking: MLflow (DagsHub)
- Deploy: Streamlit + Docker + Render

### 📌 Observação
Esta aplicação é apenas para fins educacionais e não substitui diagnóstico médico.
""")

    st.write("### 📌 Exemplo de uso")

    st.code("""
{
    "pregnancies": 2,
    "glucose": 130,
    "bloodpressure": 80,
    "skinthickness": 25,
    "insulin": 100,
    "bmi": 30.5,
    "diabetespedigreefunction": 0.6,
    "age": 40
}
""")