# Diabetes Detection API

API para predição de diabetes usando Machine Learning com relatórios diagnósticos explicativos gerados por LLM.

## 📚 Tech Challenger 1

Os scripts e notebooks do **Tech Challenger 1** estão localizados na pasta `jupyter/tech-challenger-1/`:

- `Diabetes.ipynb` - Análise exploratória e treinamento do modelo
- `ExtraTechChallenge.ipynb` - Análises adicionais
- `script.txt` - Scripts auxiliares

## 🧬 Tech Challenger 2

O script de **Algoritmo Genético (AG)** para otimização de hiperparâmetros está localizado na pasta `jupyter/tech-challenger-2/`:

- `GA_train.ipynb` - Treinamento com Algoritmo Genético para otimização de threshold e hiperparâmetros do modelo

## 🚀 Como Iniciar a API

### Opção 1: Executar Localmente (Fora do Docker)

#### macOS/Linux:

```bash
# 1. Criar e ativar ambiente virtual
python3 -m venv venv
source venv/bin/activate

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Configurar variáveis de ambiente (opcional)
# Criar arquivo .env na raiz do projeto:
# LLM_PROVIDER=ollama
# OLLAMA_HOST=http://localhost:11434
# OLLAMA_MODEL=llama3.2:1b

# 4. Iniciar a API
python -m api
```

#### Windows:

```powershell
# 1. Criar e ativar ambiente virtual
python -m venv venv
venv\Scripts\activate

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Configurar variáveis de ambiente (obrigatório)
# Criar arquivo .env na raiz do projeto:
# LLM_PROVIDER=ollama
# OLLAMA_HOST=http://localhost:11434
# OLLAMA_MODEL=llama3.2:1b

# 4. Iniciar a API
python -m api
```

A API estará disponível em: `http://localhost:8000`

### Opção 2: Executar com Docker

#### macOS/Linux:

```bash
# 1. Criar arquivo .env na raiz (obrigatório)
# LLM_PROVIDER=ollama
# OLLAMA_HOST=http://localhost:11434
# OLLAMA_MODEL=llama3.2:1b
# OPENAI_API_KEY=your_key_here
# OPENAI_MODEL=gpt-4o-mini

# 2. Construir e iniciar o container
docker compose up -d

# 3. Ver logs
docker compose logs -f diabetes-api

# 4. Parar o container
docker compose down
```

#### Windows:

```powershell
# 1. Criar arquivo .env na raiz (obrigatório)
# LLM_PROVIDER=ollama
# OLLAMA_HOST=http://localhost:11434
# OLLAMA_MODEL=llama3.2:1b
# OPENAI_API_KEY=your_key_here
# OPENAI_MODEL=gpt-4o-mini

# 2. Construir e iniciar o container
docker compose up -d

# 3. Ver logs
docker compose logs -f diabetes-api

# 4. Parar o container
docker compose down
```

A API estará disponível em: `http://localhost:8000`

## 📋 Endpoints Disponíveis

- `GET /health` - Health check
- `POST /diagnostic/invoke` - Relatório diagnóstico completo (predição + explicação LLM)
- `POST /diagnostic/stream` - Relatório diagnóstico em streaming

## 🔧 Variáveis de Ambiente

| Variável | Descrição | Padrão |
|----------|-----------|--------|
| `LLM_PROVIDER` | Provedor LLM (`ollama` ou `openai`) | `ollama` |
| `OLLAMA_HOST` | URL do servidor Ollama | `http://localhost:11434` |
| `OLLAMA_MODEL` | Modelo Ollama a ser usado | `llama3.2:1b` |
| `OPENAI_API_KEY` | Chave da API OpenAI | - |
| `OPENAI_MODEL` | Modelo OpenAI a ser usado | `gpt-4o-mini` |

## 📖 Documentação da API

Após iniciar a API, acesse a documentação interativa:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
