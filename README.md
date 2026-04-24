# 🩺 Predição de Diabetes — MLOps Completo

App de predição de risco de diabetes com rastreamento completo de experimentos via DagsHub + MLflow, containerizado com Docker e publicado no Render.

Projeto didático da aula de Containers, Docker, ML, MLflow e Streamlit.

Fluxo completo
Dados (CSV) → Pré-processamento → Experimentos (MLflow) → DagsHub (comparar) → Promover modelo → Streamlit → Docker → Render
Configuração inicial
1. Clonar e instalar dependências
git clone https://github.com/SEU_USUARIO/diabetes-pred.git
cd diabetes-app
pip install -r requirements.txt
2. Criar o arquivo .env com suas credenciais do DagsHub
cp .env.example .env
# Edite o .env com seu usuário, repositório e token do DagsHub

Obtenha o token em: dagshub.com → User Settings → Tokens → New Token

Dataset

Dataset: Pima Indians Diabetes

Variáveis clínicas:
Glucose
BloodPressure
SkinThickness
Insulin
BMI
Age
Pregnancies
DiabetesPedigreeFunction
Target:
0 → não diabético
1 → diabético
Pré-processamento aplicado
Substituição de valores inválidos (0 → NaN) nas colunas:
Glucose, BloodPressure, SkinThickness, Insulin, BMI
Imputação de valores ausentes com mediana
Split treino/teste com estratificação
Executando os experimentos
Experimento 1 — baseline
C = 1.0
RUN_NAME = "exp-1-baseline"
python src/train.py
Experimento 2 — maior regularização
C = 0.1
RUN_NAME = "exp-2-regularizado"
python src/train.py
Experimento 3 — modelo alternativo

Substituir no train.py:

from sklearn.ensemble import RandomForestClassifier

("clf", RandomForestClassifier(n_estimators=100))
RUN_NAME = "exp-3-random-forest"
python src/train.py
Comparando no DagsHub
Acesse dagshub.com/SEU_USUARIO/diabetes-app
Vá na aba Experiments
Compare os runs pelas métricas:
roc_auc ⭐
f1
accuracy
Identifique o melhor modelo
Baixar o melhor modelo para uso local
python src/evaluate.py
# Gera o arquivo model.pkl
Rodar o app localmente
streamlit run app.py
# Acesse: http://localhost:8501
Rodar com Docker (local)
docker build \
  --build-arg DAGSHUB_USER=seu_usuario \
  --build-arg DAGSHUB_REPO=diabetes-app \
  --build-arg DAGSHUB_TOKEN=seu_token \
  -t diabetes-app .

docker run -p 8501:8501 diabetes-app
Deploy no Render
Acesse https://render.com
 → New → Web Service
Conecte o repositório GitHub
Runtime: Docker | Port: 8501
Adicione as variáveis de ambiente:
DAGSHUB_USER
DAGSHUB_REPO
DAGSHUB_TOKEN
Clique em Create Web Service

O Render usa o Dockerfile — o evaluate.py baixa automaticamente o modelo mais atual do DagsHub durante o build.

Estrutura do projeto
diabetes-app/
├── data/
│   └── diabetes.csv
├── src/
│   ├── __init__.py
│   ├── prepare_data.py      # Limpeza (zeros → NaN), imputação e split
│   ├── train.py             # Treinamento + MLflow logging
│   └── evaluate.py          # Baixa modelo do DagsHub e salva localmente
├── app.py                   # Interface Streamlit (inputs clínicos)
├── requirements.txt
├── Dockerfile
├── .env.example
├── .gitignore
├── .dockerignore
└── README.md
Interface do app

O usuário insere dados via sliders:

Glicose
Pressão arterial
BMI
Insulina
Idade
etc.

O modelo retorna:

📊 Probabilidade de diabetes
⚠️ Classificação: alto risco / baixo risco
Métricas utilizadas
Accuracy
F1-score
ROC-AUC ⭐
Aviso

⚠️ Este sistema é apenas educacional e não substitui diagnóstico médico profissional.