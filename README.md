# ItauHack News Pipeline

Este repositório contém utilitários para transformar notícias de mercado em eventos estruturados e validar suas previsões contra movimentos de preço.

## Requisitos

- Python 3.9+
- Dependências opcionais:
  - [`pyarrow`](https://pypi.org/project/pyarrow/) ou [`pandas`](https://pypi.org/project/pandas/) para leitura de arquivos Parquet.
- [`pandas`](https://pypi.org/project/pandas/) para validação e geração de tabelas.
- [`openpyxl`](https://pypi.org/project/openpyxl/) para leitura dos históricos em Excel (`Hist_BDRs.xlsx` e `Hist_Origem_BDRs.xlsx`).
- [`matplotlib`](https://pypi.org/project/matplotlib/) e [`seaborn`](https://pypi.org/project/seaborn/) para gráficos de validação.

Instale todas as dependências com:

```bash
pip install pandas openpyxl pyarrow matplotlib seaborn
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

### Como os preços são encontrados

- A validação consome exclusivamente os históricos locais `data/Hist_BDRs.xlsx` e `data/Hist_Origem_BDRs.xlsx`.
- Cada aba corresponde a um ticker BDR (ex.: `AAPL34`). Somente esses ativos serão avaliados.
- O código mapeia automaticamente sinônimos comuns das notícias (ex.: `GOOGL`, `GOOG`, `GOGL35`) para o BDR correspondente (`GOGL34`).
- Caso uma notícia traga um ticker fora da cobertura das planilhas, o evento é ignorado e registrado em `price_availability.csv` como `ticker_not_covered`.

O script gera:

- `detailed_results.parquet` (ou `.csv` em fallback) com cada notícia avaliada por horizonte.
- `accuracy_summary.csv` com acurácia, acertos e neutros por horizonte.
- `confusion_matrix.csv` com o cruzamento movimento esperado x realizado.
- `accuracy_by_horizon.png` e `confusion_heatmap.png` com visualizações.
- `price_availability.csv` com o status das tentativas de busca de preço e `price_availability.png` com o gráfico de sucesso/fracasso.
- `multi_news_summary.csv` e `multi_news_accuracy.png` destacando o desempenho em dias com múltiplas notícias para o mesmo ativo.
- `rolling_window_summary.csv` e `rolling_window_accuracy.png` avaliando o efeito combinado das notícias dos últimos 3 e 5 dias sobre os próximos 1, 3 e 5 dias.

## Diagnóstico

Use `--log-level DEBUG` em qualquer script para rastrear as decisões da heurística ou mensagens detalhadas de download de preços.

Como a validação utiliza apenas os históricos locais em Excel, nenhuma conexão externa é necessária. Verifique `price_availability.csv` para identificar quais tickers foram ignorados por falta de cobertura nas planilhas ou por ausência de dados no intervalo solicitado.

#### Sobre o log "fora da cobertura disponível"

Cada aba dos arquivos `Hist_BDRs.xlsx` e `Hist_Origem_BDRs.xlsx` possui uma janela fixa de datas. Quando uma notícia é mais recente do que a última linha disponível nessas planilhas (ou anterior ao início da série), o script não consegue montar a janela de 1/3/5 dias necessária para calcular os retornos. Nesses casos você verá logs como:

```
INFO:src.validation:Evento <ID> (<ticker>) fora da cobertura disponível para <BDR>: <data_inicial> a <data_final>
```

Isso significa que o evento foi ignorado porque não existem cotações no Excel para o período solicitado. Para remover o aviso, atualize as planilhas com dados mais recentes (ou históricos mais antigos, conforme o caso) e execute novamente a validação.
