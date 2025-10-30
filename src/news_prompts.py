"""Prompt templates used across the news analysis pipeline."""

from __future__ import annotations

from textwrap import dedent


RELEVANCE_PROMPT = dedent(
    """
    Você é um analista de mercado extremamente criterioso. Classifique a notícia
    abaixo em uma das seguintes categorias: `market_moving`, `fluff_marketing`
    ou `irrelevant`.

    Regras:
    - `market_moving`: eventos que podem influenciar lucros, produção,
      demissões, fusões, processos, novas diretrizes e qualquer informação que
      afete diretamente o desempenho financeiro ou operacional da empresa.
    - `fluff_marketing`: campanhas promocionais, matérias institucionais,
      premiações, iniciativas sociais ou entrevistas genéricas.
    - `irrelevant`: notícias sem relação com o desempenho ou com a empresa
      citada.

    Responda somente com o rótulo final sem explicações adicionais.

    Notícia:
    {news}
    """
).strip()


STRUCTURED_EVENT_PROMPT = dedent(
    """
    Você é um analista financeiro. Extraia as informações da notícia a seguir e
    produza um JSON com o seguinte formato (chaves obrigatórias):

    {
      "evento_tipo": string com o melhor resumo do evento,
      "sentimento_geral": "positivo" | "negativo" | "neutro" | "misto",
      "impacto": {
        "nota": número inteiro de 1 a 10 sobre o impacto potencial no preço,
        "justificativa": breve justificativa textual (máx. 40 palavras)
      },
      "metricas": [
        {
          "metrica": nome da métrica (ex: receita, EPS, guidance, produção),
          "valor": número ou null se ausente,
          "expectativa": número ou null se mencionada,
          "resultado": "beat" | "miss" | "inline" | "desconhecido"
        }
      ]
    }

    Siga etapas de raciocínio internamente e responda SOMENTE com o JSON final.

    Notícia:
    {news}
    """
).strip()
