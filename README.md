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
pip install pandas pyarrow scikit-learn xgboost yfinance openai python-dateutil matplotlib seaborn
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

## Uso Básico (via Python)

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

## Execução em Batch pela linha de comando

O módulo pode ser executado diretamente para rodar todo o fluxo (carregar dados,
treinar o XGBoost para `ret_1d`, `ret_3d`, `ret_5d`, gerar previsões, tabelas e
gráficos). Exemplo:

```bash
python -m src.news_impact_analysis \
  --ticker AAPL \
  --start 2023-01-01 \
  --end 2023-06-30 \
  --data-dir data \
  --output-dir reports/aapl \
  --horizons ret_1d,ret_3d,ret_5d
```

Principais argumentos:

- `--ticker`: código do ativo.
- `--start` / `--end`: (opcionais) delimitam o período considerado.
- `--horizons`: lista de horizontes de retorno (default `ret_1d,ret_3d,ret_5d`).
- `--no-plots`: pule a geração de gráficos, útil para servidores sem interface.
- `--top-features`: controla quantas features aparecem no gráfico de
  importância.
- `--llm-model` e `--openai-api-key`: configuram o modelo de linguagem
  utilizado. Sem chave, o sistema cai nas heurísticas.

Ao final, o CLI imprime o caminho de cada artefato gerado e um resumo numérico
das métricas de previsão.

## Saídas Geradas

Todos os arquivos são gravados dentro do diretório informado em `--output-dir`
(ou `reports/` por padrão), organizados nas pastas `tables/` e `figures/`.

### Tabelas

- `<ticker>_<inicio>_<fim>_dataset.parquet`: features pré-modelo, incluindo
  `impact_score`, métricas do evento, atributos de disseminação e retornos
  realizados.
- `<ticker>_<inicio>_<fim>_scored.parquet`: mesmas colunas do dataset com os
  campos `pred_ret_1d`, `pred_ret_3d`, `pred_ret_5d` adicionados (retornos em
  formato decimal, ex.: 0.025 = 2,5%).
- `<ticker>_<inicio>_<fim>_summary.csv`: tabela agregada com métricas por
  horizonte (`media_retorno_real_pct`, `media_retorno_previsto_pct`, `mae_pct`,
  `rmse_pct` e `correlacao`).

### Gráficos

- `<ticker>_<inicio>_<fim>_impact_vs_returns.png`: regressões mostrando a
  relação entre a nota de impacto (1–10) e o retorno realizado em cada horizonte.
- `<ticker>_<inicio>_<fim>_predicted_vs_real.png`: séries temporais comparando o
  retorno realizado vs. previsto pelo XGBoost (em pontos percentuais).
- `<ticker>_<inicio>_<fim>_feature_importance_<horizonte>.png`: barras com as
  principais variáveis para cada horizonte previsto.

## Execução em Batch (via código)

Para automatizar pela API Python, você pode reaproveitar o método
`run_full_analysis`:

```python
from pathlib import Path
from datetime import datetime
from src.news_impact_analysis import build_default_analyzer

analyzer = build_default_analyzer(Path("data"))
artifacts = analyzer.run_full_analysis(
    ticker="MSFT",
    start=datetime(2024, 1, 1),
    end=datetime(2024, 6, 30),
    output_dir=Path("reports/msft"),
)

print(artifacts.summary)
```

O objeto `AnalysisArtifacts` contém os DataFrames em memória (`dataset`,
`scored`, `summary`) e os caminhos para as tabelas e figuras salvas.

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
