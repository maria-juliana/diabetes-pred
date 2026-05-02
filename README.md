# 🩺 Diabetes Risk Prediction — MLOps Pipeline

Aplicação de Machine Learning para predição de risco de diabetes com pipeline completo de MLOps usando MLflow + DagsHub + Docker + Streamlit + Render.

🚀 Funcionalidades
- Predição de risco de diabetes (baixo / médio / alto)
- Probabilidade (%)
- Interface interativa com Streamlit
- Rastreamento de experimentos (MLflow + DagsHub)
- Deploy em produção (Render)

📊 Dataset
Pima Indians Diabetes Dataset
Classificação binária: 0 (não) / 1 (sim)
Fonte: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database  

🔄 Pipeline
Data → Preprocess → Train → MLflow → DagsHub → Evaluate → Streamlit → Deploy

🛠️ Como rodar localmente
git clone https://github.com/SEU_USUARIO/diabetes-pred.git
cd diabetes-pred
pip install -r requirements.txt

->Treinar modelo
python -m src.train

->Baixar modelo
python -m src.evaluate

->Rodar app
streamlit run app.py

🐳 Docker
docker build -t diabetes-app .
docker run -p 8501:8501 diabetes-app

☁️ Deploy
Deploy via Render usando Docker.

📂 Estrutura
src/
app.py
Dockerfile
requirements.txt

⚠️ Aviso
Projeto educacional. Não substitui diagnóstico médico.

🧠 Tecnologias
- scikit-learn
- Streamlit
- MLflow
- DagsHub
- Docker
- Render