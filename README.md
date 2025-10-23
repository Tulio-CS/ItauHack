# News Impact Analyzer

Este projeto adiciona uma pipeline completa para estimar o impacto de notícias
financeiras (bases `investing_news*.parquet`) sobre o preço de um ticker na
bolsa. O fluxo implementa os três upgrades solicitados:

1. **Filtro de Relevância com LLM** – classifica cada notícia como
   *Market-Moving*, *Fluff/Marketing* ou *Irrelevante* antes de seguir com a
   análise.
2. **Extração Estruturada + Cadeia de Pensamento** – gera um JSON rico com o
   tipo de evento, métricas (beat/miss) e um *impact score* (1-10) justificado,
   transformando texto em *features* quantitativas.
3. **Fator de Disseminação (FinGPT)** – clusteriza as notícias em janelas de
   tempo para capturar o buzz informacional (número de clusters ativos, tamanho
   do maior cluster, velocidade e sentimento ponderado).

Os dados são combinados com séries históricas de preços (via Yahoo Finance) e a
relação entre features e retornos futuros (1d/3d/5d) é estimada com um modelo
[XGBoost](https://xgboost.readthedocs.io/).

## Estrutura do Código

O código principal está em `src/news_impact_analysis.py` e expõe a classe
`NewsImpactAnalyzer`. Os componentes principais são:

- `NewsLoader`: consolida e normaliza as três bases parquet.
- `LLMAnalyzer`: encapsula as chamadas ao LLM para filtragem, extração de
  eventos e nota de impacto (com *fallback* heurístico quando o LLM não está
  disponível).
- `DisseminationFeatureBuilder`: calcula os atributos de disseminação usando
  TF-IDF + `MiniBatchKMeans`.
- `PriceFetcher`: baixa automaticamente as cotações pelo `yfinance`.
- `XGBoostImpactModel`: treinamento e predição dos retornos em diferentes
  horizontes.

## Requisitos

Instale as dependências principais antes de rodar os exemplos:

```bash
pip install pandas pyarrow scikit-learn xgboost yfinance openai python-dateutil
```

> ⚠️ Caso sua rede bloqueie o PyPI, configure o proxy corporativo ou faça o
> download offline dos *wheels*. O código tem *fallbacks* heurísticos para
> cenários sem LLM, permitindo testes básicos mesmo sem a API.

## Configurando a Chave de API do LLM

Defina a variável de ambiente `OPENAI_API_KEY` (ou passe `openai_api_key`
explicitamente ao construir o `NewsImpactAnalyzer`). Exemplo:

```bash
export OPENAI_API_KEY="sk-..."
```

## Uso Básico

```python
from pathlib import Path
from datetime import datetime
from src.news_impact_analysis import build_default_analyzer

analyzer = build_default_analyzer(Path("data"))

dataset = analyzer.prepare_dataset(
    ticker="AAPL",
    start=datetime(2023, 1, 1),
    end=datetime(2023, 12, 31),
)

# Treine o modelo usando o retorno de 3 dias como alvo
analyzer.train_model(dataset, horizon="ret_3d")

# Gere previsões (exemplo: retorna um DataFrame com features + `pred_ret_1d`)
scored = analyzer.score(dataset)
print(scored[["published_at", "impact_score", "ret_3d", "pred_ret_1d"]].head())
```

### Output do `prepare_dataset`

O DataFrame retornado contém, entre outras, as colunas:

- `evento_tipo`, `metricas`, `sentimento_geral`: informações estruturadas do
  evento.
- `impact_score`, `impact_sentiment`, `impact_justificativa`: resultado do
  prompt de cadeia de pensamento.
- `numero_clusters_ativos`, `tamanho_maior_cluster`, `velocidade_cluster`,
  `sentimento_ponderado_cluster`: features de disseminação.
- `ret_1d`, `ret_3d`, `ret_5d`: retornos realizados após 1, 3 e 5 dias.

## Execução em Batch

Para analisar outro ticker ou ampliar a janela temporal, basta chamar novamente
`prepare_dataset` e `train_model`. Reaproveite o modelo treinado para *score*
rápido de novas notícias:

```python
new_dataset = analyzer.prepare_dataset(ticker="MSFT", start=datetime(2024, 1, 1))
predicted = analyzer.score(new_dataset)
```

## Estratégias de Cache

- **Cache de respostas do LLM**: persista as respostas do `LLMAnalyzer` em disco
  (ex.: SQLite ou JSON) para evitar chamadas repetidas.
- **Cache de preços**: use `yfinance` com `session` customizado ou salve as
  séries locais para acelerar execuções subsequentes.

## Limitações e Próximos Passos

- Ajuste os prompts para cada idioma/setor conforme necessário.
- Adapte o modelo para classificação (ex.: prever direção do retorno) se for
  mais adequado ao seu caso.
- Enriquecer as features com embeddings (quando disponíveis) pode melhorar o
  clustering e a performance do XGBoost.

## Licença

Uso interno para o hackathon Itaú. Ajuste a licença conforme sua necessidade.
