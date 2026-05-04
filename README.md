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

A pasta Notebooks possui um notebook .ipynb com uma breve análise exploratória dos dados.

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

- Parâmetros
- Métricas
- Artefatos

Foram realizados múltiplos experimentos variando hiperparâmetros.

Exemplo dos parâmetros de treinamento configurados em `src/train.py`:

```python
C = 0.1
RUN_NAME = "exp-2-baseline"
CLASS_WEIGHT = "balanced"
```

Para testar novos experimentos, altere os valores diretamente em `src/train.py` e execute o script novamente. Mudar `RUN_NAME` ajuda a diferenciar cada execução no MLflow, enquanto `C` e `CLASS_WEIGHT` controlam a regularização e o tratamento de classes desbalanceadas.

### 5. Model Registry e teste - train.py/predict.py

O melhor modelo foi registrado no MLflow (via DagsHub) e promovido para Production, permitindo seu uso direto na aplicação.

🤖 Modelo

Algoritmo: Regressão Logística
Tipo: Classificação binária
Métricas (aproximadas)
Accuracy: ~70-72%
F1-score: ~0.54-0.63

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

Criar conta em (indicado logar com GitHub):
-Kaggle;
-DagsHub;
-Supabase;
-Render (obrigatório logar com GitHub).

Baixar Docker

1. Criar tabela no banco do Supabase

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

2. Fazer fork do repositório da Maria no GitHub

No GitHub, abra o repositório original e clique em `Fork` para criar uma cópia na sua conta:
https://github.com/maria-juliana/diabetes-pred

3. Clonar seu fork
```bash
git clone https://github.com/seu_usuario/diabetes-pred.git
cd diabetes-pred
```

4. Criar e ativar ambiente virtual

Linux/macOS:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Windows (PowerShell):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

5. Configurar variáveis de ambiente
```bat
copy .env.example .env
```

→ Em caso de conflito entre DagsHub e Supabase, tentar dar update no pip.

Preencha no `.env`:

- Cópia da URL do projeto no Supabase
- Token de acesso: app.supabase.com → Project Settings → API Keys → Legacy anon → chave anon/public


```env
SUPABASE_URL=sua_url_aqui
SUPABASE_KEY=sua_key_aqui
```
- Usuário no DagsHub
- Nome do repositório
- Token de acesso: dagshub.com → User Settings → Tokens → New Token
```env

DAGSHUB_USER=seu_usuario_aqui
DAGSHUB_REPO=seu_repositorio_aqui
DAGSHUB_TOKEN=seu_token_aqui
```

- `KAGGLE_USERNAME` e `KAGGLE_KEY` (Kaggle > Account > API > Create New Token)

6. Configurar remote do DVC no DagsHub (sua conta)
```bash
dvc remote add -d origin https://dagshub.com/seu_usuario/seu_repositorio.dvc
dvc remote modify --local origin auth basic
dvc remote modify --local origin user seu_usuario
dvc remote modify --local origin password seu_token
```

Se você clonou o projeto de outra pessoa e o `origin` já veio configurado para outra conta, rode:
```bash
dvc remote remove origin
dvc remote add -d origin https://dagshub.com/seu_usuario/seu_repositorio.dvc
dvc remote modify --local origin auth basic
dvc remote modify --local origin user seu_usuario
dvc remote modify --local origin password seu_token
```

7. Validar ambiente antes do pipeline
```bash
dvc repro validate
```

8. Rodar pipeline completo com DVC
```bash
dvc repro
``` 

1. Publicar dados versionados no DagsHub
```bash
dvc push
```

Para testar modelos diferentes em `src/train.py`, altere `C` e `RUN_NAME` e reexecute somente o stage de treino:
```bash
dvc repro --single-item train --force
``` 

10. Configurar alias `production` no Model Registry (obrigatório antes da aplicação)

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

11. Rodar aplicação
```bash
streamlit run app.py
```

Opcional: upload manual da pasta `data/` no DagsHub (não substitui `dvc push`):

```bash
dagshub upload seu_usuario/seu_repositorio data/ data/
```
Automaticamente ocorrerá um redirecionamento para a página do DagsHub solicitando autorização.

🐳 Executando com Docker 

```
docker build `
--build-arg DAGSHUB_USER=seu_usuario `
--build-arg DAGSHUB_REPO=seu_repositorio `
--build-arg DAGSHUB_TOKEN=seu_token `
-t diabetes-app .
``` 
```
docker run -p 8501:8501 diabetes-app
```

🌐 Deploy com Render

Use o fork criado na etapa 2 (`https://github.com/seu_usuario/diabetes-pred`).

1. Acesse render.com e faça login com o GitHub;
2. Clique em New → Web Service;
3. Conecte o seu fork;
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
Alexandra Amaro, Carlos Alberto Mota, Edney Farias, Maria Juliana Monte e Sidney Melo.

Para a matéria Infraestrutura em Nuvem para Projetos com Ciência dos Dados ministrada pelo Prof. Dr. Fábio Santos da Silva (fssilva@uea.edu.br)
do curso de Pós-Graduação Lato Sensu em Ciência de Dados da Universidade do Estado do Amazonas (UEA).
