# signal-enrichment-agent

> Enriquecimento automático em lote de dados textuais usando LLMs (GPT-4o / Gemini).  
> Lê da camada **Silver** do DuckDB, enriquece com IA e escreve na camada **Gold**.

---

## O que este projeto resolve

Dados brutos raramente chegam com contexto suficiente para análise. Uma tabela com milhares de avaliações de produtos não tem classificação de sentimento, categoria ou entidades identificadas. Classificar tudo manualmente levaria semanas.

O `signal-enrichment-agent` resolve isso com três agentes especializados que rodam em lote, de forma controlada e com custo rastreado:

| Campo enriquecido | Exemplo de output |
|---|---|
| `sentiment` | `positive` / `negative` / `neutral` |
| `sentiment_confidence` | `0.97` |
| `category` | `Electronics` |
| `subcategory` | `Headphones` |
| `entities_brands` | `["Apple", "Samsung"]` |
| `entities_locations` | `["Shopping Iguatemi"]` |
| `entities_persons` | `[]` |

---

## Arquitetura

```
DuckDB Silver (texto bruto)
        ↓
   duckdb_reader.py          ← lê apenas registros pendentes
        ↓
   batch_processor.py        ← divide em lotes, orquestra os agentes
        ↓
┌─────────────────────────────────────┐
│  Por registro:                      │
│  SentimentAgent  → sentiment        │
│  CategoryAgent   → category         │
│  EntityAgent     → entities         │
│  CostTracker     → log de custo     │
│  Checkpoint      → posição salva    │
└─────────────────────────────────────┘
        ↓
   duckdb_writer.py          ← upsert idempotente no Gold
        ↓
DuckDB Gold (registros enriquecidos)
```

### Estrutura de diretórios

```
signal-enrichment-agent/
│
├── agents/
│   ├── base_agent.py         # contrato ABC + retry com backoff exponencial
│   ├── sentiment_agent.py    # classifica sentimento (positive/negative/neutral)
│   ├── category_agent.py     # classifica categoria + subcategoria
│   └── entity_agent.py       # extrai marcas, locais e pessoas
│
├── orchestration/
│   ├── batch_processor.py    # orquestra agentes em lote
│   ├── cost_tracker.py       # contabiliza tokens e custo em USD
│   ├── checkpoint.py         # persistência atômica de progresso
│   └── rate_limiter.py       # dual-bucket sliding window (RPM + TPM)
│
├── prompts/
│   └── v1/                   # templates versionados — editáveis sem tocar no código
│       ├── sentiment.txt
│       ├── category.txt
│       └── entity.txt
│
├── db_io/
│   ├── duckdb_reader.py      # lê Silver, filtra registros pendentes
│   └── duckdb_writer.py      # upsert idempotente no Gold
│
├── tests/
│   ├── test_agents.py        # 20 testes unitários, zero chamadas reais à API
│   └── fixtures/
│       └── sample_reviews.json
│
├── notebooks/
│   └── exploration.ipynb     # análise exploratória dos dados enriquecidos
│
├── .github/workflows/
│   └── enrichment.yml        # CI/CD — dispara após pipeline de ingestão
│
├── .env.example
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Pré-requisitos

- Python 3.11+
- Chave de API da OpenAI (ou Google para Gemini)
- DuckDB com camada Silver populada (output do projeto de ingestão)

### 2. Instalação

```bash
git clone https://github.com/seu-usuario/signal-enrichment-agent.git
cd signal-enrichment-agent

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Configuração

```bash
cp .env.example .env
# Edite .env com sua OPENAI_API_KEY e o caminho do DuckDB
```

### 4. Rodar os testes

```bash
pytest tests/ -v
```

Saída esperada: **20 passed** — sem chamadas reais à API.

### 5. Executar o pipeline

```python
from pathlib import Path
from agents.base_agent import AgentConfig
from agents.sentiment_agent import SentimentAgent
from agents.category_agent import CategoryAgent
from agents.entity_agent import EntityAgent
from db_io.duckdb_reader import DuckDBReader
from db_io.duckdb_writer import DuckDBWriter
from orchestration.batch_processor import BatchProcessor, BatchProcessorConfig
from orchestration.rate_limiter import RateLimiter

db_path = Path("data/warehouse.duckdb")
config = BatchProcessorConfig(batch_size=50, budget_usd=5.0)
agent_config = AgentConfig(model="gpt-4o", max_retries=3)

processor = BatchProcessor(
    config=config,
    reader=DuckDBReader(db_path=db_path),
    writer=DuckDBWriter(db_path=db_path),
    sentiment_agent=SentimentAgent.from_env(agent_config),
    category_agent=CategoryAgent.from_env(agent_config),
    entity_agent=EntityAgent.from_env(agent_config),
    rate_limiter=RateLimiter(rpm=500, tpm=30_000),
)

summary = processor.run()
print(summary)
# {'run_id': 'a1b2c3d4', 'records_processed': 10000, 'total_cost_usd': 2.84, ...}
```

---

## Input e Output

**Input — registro Silver:**

```json
{
  "review_id": "rev_001",
  "product_id": "prod_123",
  "review_text": "Produto excelente, chegou rápido e bem embalado.",
  "date": "2024-06-01"
}
```

**Output — registro Gold enriquecido:**

```json
{
  "review_id": "rev_001",
  "product_id": "prod_123",
  "review_text": "Produto excelente, chegou rápido e bem embalado.",
  "date": "2024-06-01",
  "sentiment": "positive",
  "sentiment_confidence": 0.97,
  "category": "E-commerce",
  "subcategory": "Delivery",
  "entities_brands": [],
  "entities_locations": [],
  "entities_persons": [],
  "enriched_at": "2024-06-02T06:00:00+00:00",
  "model_used": "gpt-4o",
  "tokens_used": 142,
  "enrichment_cost_usd": 0.000284,
  "enrichment_error": false
}
```

---

## Decisões de design

### Por que três agentes separados?

**Custo e resiliência.** Você pode rodar apenas `SentimentAgent` sem pagar por category e entity. Se o `EntityAgent` falhar em um lote, os outros dois já processaram e estão salvos.

### Como o checkpoint funciona?

O `Checkpoint` usa `os.replace()` — a única operação atômica no POSIX — para nunca deixar um arquivo corrompido em disco. Se o pipeline de 10.000 registros falhar no registro 7.342, basta re-executar: o processamento retoma do lote exato onde parou, sem repagar por registros já feitos.

### Como o rate limiter funciona?

`RateLimiter` implementa um **dual-bucket sliding window**: um bucket para RPM (requisições por minuto) e outro para TPM (tokens por minuto), exatamente como a OpenAI os aplica de forma independente. A janela deslizante evita o *thundering herd* que uma janela fixa causaria quando o período reseta.

### Por que os prompts ficam em `.txt` e não no código Python?

Prompts mudam com frequência — ajustes de framing, adição de exemplos, mudanças de formato. Mantê-los em arquivos versionados (`prompts/v1/`) permite que qualquer pessoa da equipe itere sem precisar abrir código Python. Quando os prompts evoluírem, crie `prompts/v2/` sem apagar o `v1/`.

### Por que `INSERT OR REPLACE` no writer?

Garante **idempotência**: rodar o pipeline duas vezes sobre os mesmos registros os atualiza, sem duplicar linhas. Isso é essencial para re-enriquecimento (quando um modelo mais novo é adotado) e para retomadas após falha.

---

## Controle de custo

Cada chamada à API é registrada em `logs/cost.jsonl`:

```json
{
  "timestamp": "2024-06-02T06:00:01+00:00",
  "agent": "SentimentAgent",
  "model": "gpt-4o",
  "record_id": "rev_001",
  "input_tokens": 85,
  "output_tokens": 57,
  "total_tokens": 142,
  "cost_usd": 0.000284,
  "batch_id": "a1b2c3d4"
}
```

O `CostTracker` mantém totais acumulados em memória e por agente/modelo. Configure `ENRICHMENT_BUDGET_USD` no `.env` para receber alertas quando o custo acumulado ultrapassar o limite.

**Estimativa de custo com GPT-4o (gpt-4o, ~150 tokens/registro):**

| Registros | 3 agentes | Custo estimado |
|---|---|---|
| 1.000 | 3.000 chamadas | ~$0,07 |
| 10.000 | 30.000 chamadas | ~$0,70 |
| 100.000 | 300.000 chamadas | ~$7,00 |

---

## Adicionando um novo agente

1. Crie `agents/meu_agente.py` herdando de `BaseAgent[MinhaOutput]`
2. Implemente `_build_prompt()`, `_call_llm()`, `_parse_response()`, `validate()` e `enrich()`
3. Adicione o prompt em `prompts/v1/meu_agente.txt`
4. Passe o agente ao `BatchProcessor` via o parâmetro correspondente
5. Escreva testes em `tests/test_agents.py` mockando o cliente

---

## CI/CD

O workflow `.github/workflows/enrichment.yml` é disparado automaticamente quando o pipeline de ingestão (Data Ingress Framework) termina com sucesso. A cadeia completa:

```
Ingestão roda → dados chegam no Silver → Enriquecimento roda → dados chegam no Gold
```

O workflow também pode ser disparado manualmente via `workflow_dispatch` com parâmetros de `batch_size`, `run_id` e `dry_run`.

---

## Análise exploratória

Após rodar o pipeline, abra o notebook:

```bash
cd notebooks
jupyter notebook exploration.ipynb
```

O notebook cobre: distribuição de sentimentos, mix de sentimento por categoria, top entidades, evolução de custo ao longo do tempo e identificação de registros de baixa confiança para re-enriquecimento.

---

## Variáveis de ambiente

| Variável | Descrição | Default |
|---|---|---|
| `OPENAI_API_KEY` | Chave da OpenAI | — |
| `DUCKDB_PATH` | Caminho do arquivo DuckDB | `data/warehouse.duckdb` |
| `BATCH_SIZE` | Registros por lote | `50` |
| `ENRICHMENT_BUDGET_USD` | Limite de custo em USD | `5.00` |
| `RPM` | Requisições por minuto | `500` |
| `TPM` | Tokens por minuto | `30000` |
| `LOG_LEVEL` | Nível de log | `INFO` |
| `COST_LOG_PATH` | Caminho do log de custo | `logs/cost.jsonl` |
| `CHECKPOINT_DIR` | Diretório de checkpoints | `.checkpoints` |
