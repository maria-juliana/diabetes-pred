# Predição de Diabetes com Pipeline de Machine Learning (MLOps)

## Visão Geral

Este projeto implementa um pipeline completo de Machine Learning end-to-end para predição de diabetes, integrando ferramentas modernas de MLOps.

O objetivo é demonstrar um fluxo completo desde a ingestão de dados até o deploy de uma aplicação interativa, incluindo versionamento, experimentação e monitoramento de modelos.

---

## Problema

O projeto busca prever a probabilidade de um paciente desenvolver diabetes com base em variáveis clínicas, como glicose, pressão arterial, IMC e idade.

Trata-se de um problema de **classificação binária**, onde:

- `0` → Não possui diabetes  
- `1` → Possui diabetes  

O dataset é o Pima Indians Diabetes encontrado em: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database.

---

## Tecnologias Utilizadas

- **Supabase** → fonte de dados (PostgreSQL)
- **DuckDB** → processamento analítico via SQL
- **DVC + DagsHub** → versionamento de dados e pipeline
- **MLflow** → tracking e registro de modelos
- **Scikit-learn** → modelagem
- **Streamlit** → interface interativa
- **Docker + Render** → deploy

---

## Pipeline de Dados

### 1. Download do dataset - download_data.py

Download dos dados diretamente da API do Kaggle.

### 2. Ingestão (Supabase) - ingestion.py

Os dados são armazenados no Supabase e acessados via API, funcionando como fonte transacional do sistema.

### 3. Processamento (DuckDB) - preprocessing.py

O DuckDB é utilizado como engine analítica para realizar transformações via SQL:

- Substituição de valores inválidos (`0 → NULL`)
- Imputação com mediana
- Criação de features:
  - `high_glucose_flag`

Exemplo:

```sql
COALESCE(NULLIF(glucose, 0), MEDIAN(NULLIF(glucose, 0)) OVER())
```

### 3. Versionamento (DVC + DagsHub) - dvc.yaml

Os dados processados são versionados com DVC, garantindo:

Reprodutibilidade
Controle de versões
Separação entre código e dados

### 4. Treinamento (MLflow) - train.py

O modelo foi treinado utilizando Regressão Logística, com tracking de:

Parâmetros
Métricas
Artefatos

Foram realizados múltiplos experimentos variando hiperparâmetros.

### 5. Model Registry e teste - train.py/predict.py

O melhor modelo foi registrado no MLflow (via DagsHub) e promovido para Production, permitindo seu uso direto na aplicação.

🤖 Modelo

Algoritmo: Regressão Logística
Tipo: Classificação binária
Métricas (aproximadas)
Accuracy: ~70-71%
F1-score: ~0.54-0.56

🔮 Inferência

A função de inferência (predict.py) é responsável por:

Carregar o modelo do MLflow
Garantir consistência das features
Retornar:
previsão (Sim/Não)
probabilidade

💻 Aplicação (Streamlit)

A aplicação permite:

Inserção de dados clínicos via interface amigável
Visualização da probabilidade de diabetes
Classificação de risco (baixo, médio, alto)
Aba de análise exploratória dos dados

📊 Funcionalidades do App
Inputs interativos (sliders)
Previsão em tempo real
Exibição de risco
Estatísticas descritivas
Visualizações simples (EDA)

## Como Rodar o Projeto

Criar conta em (logar com github):
-DagsHub;
-Supabase;
-Render.

Baixar Docker

1. Criar tabela no banco do Supabase:

- Criação de um projeto em app.supabase.com
- No editor SQL do projeto (CTRL+E) adicionar e rodar:
```sql
create table diabetes(
id SERIAL primary key,
pregnancies integer,
glucose integer,
bloodpressure integer,
skinthickness integer,
insulin integer,
bmi float,
diabetespedigreefunction float,
age integer,
outcome integer
)
```
2. Criação de novo repositório no DagsHub

3. Clonar repositório
git clone <repo>
cd <nome repo>

4. Instalar dependências
pip install -r requirements.txt

→ Em caso de conflito entre DagsHub e Supabase, tentar dar update no pip.

5. Configurar variáveis de ambiente
copy .env.example .env

- Cópia da URL do projeto no Supabase
- Token de acesso: app.supabase.com → Project Settings → API Keys → Legacy anon → chave anon/public

SUPABASE_URL=sua_url_aqui
SUPABASE_KEY=sua_key_aqui

- Usuário no DagsHub
- Nome do repositório
- Token de acesso: dagshub.com → User Settings → Tokens → New Token

DAGSHUB_USER=seu_usuario_aqui
DAGSHUB_REPO=seu_repositorio_aqui
DAGSHUB_TOKEN=seu_token_aqui

4. Rodar pipeline completo
dvc repro

Para testagem de modelos diferentes em train.py, é preciso modificar C e RUN_NAME e rodar dvc repro a cada mudança.
→ Talvez seja necessário utilizar dvc repro --force
Para baixar o modelo com melhor resultado e seguir com o projeto, é preciso reproduzir os seguintes passos:

1. No DagsHub, clique em Go to MLflow UI (botão no canto superior direito da aba Experiments).
2. No menu lateral do MLflow, clique em Models;
3. Clique em DiabetesClassifier;
4. Você verá as versões registradas (Version 1, 2, 3...). Identifique a versão com maior F1 pelo Supabase;
5. Na linha dessa versão, vá em Aliases → Add.
6. No campo que aparecer, digite production e confirme. A versão agora aparece marcada com o alias
production.
7. Em src/predict.py deve estar assim:
```python
def load_model():
    model = mlflow.sklearn.load_model(
        "models:/DiabetesClassifier@production"
    )
    return model
```
5. Rodar aplicação
streamlit run app.py

🐳 Executando com Docker 

docker build `
--build-arg DAGSHUB_USER=seu_usuario `
--build-arg DAGSHUB_REPO=seu_repositorio `
--build-arg DAGSHUB_TOKEN=seu_token `
-t sentimento-app .

docker run -p 8501:8501 seu_repositorio

🌐 Deploy com Render

Criar um repositório no GitHub:
git init
git add .
git commit -m "o que quiser descrever sobre seu repositório"
git branch -M main
git remote add origin URL DO REPOSITÓRIO
git push -u origin main

1. Acesse render.com e faça login com o GitHub;
2. Clique em New → Web Service;
3. Conecte seu repositório;
4. Confirme que o Runtime foi detectado como Docker;
→ O Render detecta automaticamente o EXPOSE 8501 do Dockerfile;
5. Na seção Environment Variables, adicione as três variáveis:

DAGSHUB_USER=seu_usuario_aqui
DAGSHUB_REPO=seu_repositorio_aqui
DAGSHUB_TOKEN=seu_token_aqui

6. Clique em Create Web Service;
7. A URL gerada fica nesse formato: https://seu-repositorio-xxxx.onrender.com e é compartilhável para uso!

🔐 Segurança
Credenciais armazenadas em .env
.env não versionado (incluído no .gitignore)

⚠️ Observação

Este projeto é apenas para fins educacionais e não substitui diagnóstico médico.

Este projeto foi criado pelos alunos:
→ Alexandra Amaro;
→ Carlos Alberto Mota;
→ Edney Farias;
→ Maria Juliana Monte;
→ Sidney Melo.

Para a matéria Infraestrutura em Nuvem para Projetos com Ciência dos Dados ministrada pelo Prof. Dr. Fábio Santos da Silva (fssilva@uea.edu.br)
Do curso de Pós-Graduação Lato Sensu em Ciência de Dados da Universidade do Estado do Amazonas (UEA).