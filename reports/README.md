# Relatórios gerados

Os gráficos, tabelas e métricas produzidos pelo pipeline automatizado serão
armazenados nesta pasta (ou em subdiretórios).  Execute o script principal para
popular os arquivos:

```bash
python scripts/run_news_pipeline.py \
  --news-files data/investing_news.parquet data/investing_news_nacionais.parquet \
  data/investing_news_nacionais_que_faltaram.parquet \
  --output-dir reports/generated
```

Os artefatos principais incluem:

- `market_moving_news.parquet`: subconjunto filtrado pelo LLM.
- `market_moving_with_returns.parquet`: notícias com retornos de preço
  calculados automaticamente via Yahoo Finance.
- `features.parquet`: matriz de features utilizada para treinar o modelo de
  impacto no preço.
- `classification_report.json`: métricas do modelo XGBoost.
- `feature_importance.png`, `confusao_modelo.png`, `retorno_hist.png`: gráficos
  para uso na apresentação.
- `models/news_xgb.joblib`: modelo salvo para reuso em previsões futuras.
