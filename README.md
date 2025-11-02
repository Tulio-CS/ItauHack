# ItauHack News Pipeline

Este repositório contém utilitários para transformar notícias de mercado em eventos estruturados e validar suas previsões contra movimentos de preço.

## Requisitos

- Python 3.9+
- Dependências opcionais:
  - [`pyarrow`](https://pypi.org/project/pyarrow/) ou [`pandas`](https://pypi.org/project/pandas/) para leitura de arquivos Parquet.
  - [`pandas`](https://pypi.org/project/pandas/) para validação e geração de tabelas.
- [`matplotlib`](https://pypi.org/project/matplotlib/) e [`seaborn`](https://pypi.org/project/seaborn/) para gráficos de validação.

Instale todas as dependências com:

```bash
pip install pandas pyarrow matplotlib seaborn
```

## 1. Gerar eventos estruturados com o LLM local

O script abaixo carrega os Parquets padrão do Investing News, executa a heurística local e salva um JSONL com o texto original e o evento estruturado:

```bash
python scripts/run_local_llm.py --log-level INFO
```

Opções úteis:

- `--output`: caminho alternativo para salvar o JSONL (padrão `data/output/investing_news_structured.jsonl`).
- `--limit`: processa apenas os primeiros *N* registros (ex.: `--limit 100` para testes rápidos).
- `--files`: lista personalizada de arquivos CSV/Parquet.

## 2. Validar previsões com dados de mercado

Com o JSONL estruturado em mãos, execute a validação para calcular acurácia de sentimento versus movimento de preço em 1, 3 e 5 dias. Os resultados são gravados em `reports/validation/`.

```bash
python scripts/validate_predictions.py \
  --input data/output/investing_news_structured.jsonl \
  --output-dir reports/validation \
  --log-level INFO
```

Parâmetros importantes:

- `--neutral-threshold`: retorno absoluto considerado neutro (padrão `0.01`, ou 1%).
- `--horizons`: horizontes personalizados (ex.: `--horizons 1 2 4 7`).

O script gera:

- `detailed_results.parquet` (ou `.csv` em fallback) com cada notícia avaliada por horizonte.
- `accuracy_summary.csv` com acurácia, acertos e neutros por horizonte.
- `confusion_matrix.csv` com o cruzamento movimento esperado x realizado.
- `accuracy_by_horizon.png` e `confusion_heatmap.png` com visualizações.
- `price_availability.csv` com o status das tentativas de busca de preço e `price_availability.png` com o gráfico de sucesso/fracasso.

## Diagnóstico

Use `--log-level DEBUG` em qualquer script para rastrear as decisões da heurística ou mensagens detalhadas de download de preços.

Em ambientes sem acesso à internet, a validação emitirá avisos ao tentar baixar preços no Stooq. Rode novamente quando houver conectividade.
