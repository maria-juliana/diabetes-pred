"""
app.py — Interface Streamlit para análise de diabetes.

Execute com:
    streamlit run app.py

Ou via Docker:
    docker build --build-arg DAGSHUB_USER=... -t diabetes-app .
    docker run -p 8501:8501 diabetes-app
"""

import streamlit as st
import pandas as pd
import joblib

# ── Configuração ───────────────────────────────────────
st.set_page_config(
    page_title="Predição de Diabetes",
    page_icon="🩺",
    layout="centered",
)

# ── Carregar modelo ────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")

pipeline = load_model()

# ── Interface ──────────────────────────────────────────
st.title("🩺 Predição de Risco de Diabetes")

st.markdown(
    "Insira os dados do paciente para estimar a probabilidade de diabetes. "
    "Modelo treinado com dados clínicos e rastreado no **DagsHub**."
)

st.divider()

# ── Inputs (SLIDERS) ───────────────────────────────────
st.subheader("Dados do paciente")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.slider("Número de gestações", 0, 20, 1)
    glucose = st.slider("Glicose", 0, 200, 100)
    blood_pressure = st.slider("Pressão arterial", 0, 140, 70)
    skin_thickness = st.slider("Espessura da pele", 0, 100, 20)

with col2:
    insulin = st.slider("Insulina", 0, 900, 80)
    bmi = st.slider("BMI", 0.0, 70.0, 25.0)
    dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
    age = st.slider("Idade", 10, 100, 30)

# ── Botão de predição ──────────────────────────────────
if st.button("Prever risco", type="primary", use_container_width=True):

    # Criar dataframe com mesma estrutura do treino
    input_data = pd.DataFrame([{
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": dpf,
        "Age": age,
    }])

    # Predição
    pred = pipeline.predict(input_data)[0]
    proba = pipeline.predict_proba(input_data)[0][1]

    # Resultado
    st.divider()

    if pred == 1:
        st.error(f"⚠️ Alto risco de diabetes ({proba:.0%})")
    else:
        st.success(f"✅ Baixo risco de diabetes ({proba:.0%})")

    # Probabilidade detalhada
    st.markdown("### Probabilidade")
    st.progress(float(proba), text=f"Risco de diabetes: {proba:.0%}")

st.divider()

# ── Info ───────────────────────────────────────────────
st.caption(
    "Modelo: Logistic Regression + StandardScaler · "
    "Rastreamento: MLflow + DagsHub · "
    "Deploy: Docker + Render"
)