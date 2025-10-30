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

    Responda **exclusivamente** com um JSON válido no formato:
    {{"relevance": "market_moving"}}
    (use um dos três rótulos descritos).

    Notícia:
    {news}
    """
).strip()


RELEVANCE_RETRY_PROMPT = dedent(
    """
    A resposta anterior não estava em JSON válido ou não usou o rótulo correto.
    Gere novamente UM JSON válido no formato:
    {{"relevance": "market_moving"}}
    (o valor deve ser `market_moving`, `fluff_marketing` ou `irrelevant`).

    Notícia:
    {news}

    Resposta anterior:
    {previous_response}
    """
).strip()


STRUCTURED_EVENT_PROMPT = dedent(
    """
    Você é um analista financeiro. Extraia as informações da notícia a seguir e
    produza um JSON com o seguinte formato (chaves obrigatórias):

    {{
      "evento_tipo": string com o melhor resumo do evento,
      "sentimento_geral": "positivo" | "negativo" | "neutro" | "misto",
      "impacto": {{
        "nota": número inteiro de 1 a 10 sobre o impacto potencial no preço,
        "justificativa": breve justificativa textual (máx. 40 palavras)
      }},
      "metricas": [
        {{
          "metrica": nome da métrica (ex: receita, EPS, guidance, produção),
          "valor": número ou null se ausente,
          "expectativa": número ou null se mencionada,
          "resultado": "beat" | "miss" | "inline" | "desconhecido"
        }}
      ]
    }}

    Siga etapas de raciocínio internamente e responda **SOMENTE** com o JSON
    final.

    Notícia:
    {news}
    """
).strip()


STRUCTURED_EVENT_RETRY_PROMPT = dedent(
    """
    A resposta anterior não estava em JSON válido. Gere novamente um JSON
    seguindo exatamente o formato indicado abaixo (todas as chaves são
    obrigatórias):

    {{
      "evento_tipo": string,
      "sentimento_geral": "positivo" | "negativo" | "neutro" | "misto",
      "impacto": {{"nota": inteiro 1-10, "justificativa": string}},
      "metricas": [
        {{
          "metrica": string,
          "valor": número ou null,
          "expectativa": número ou null,
          "resultado": "beat" | "miss" | "inline" | "desconhecido"
        }}
      ]
    }}

    Produza apenas o JSON final, sem comentários.

    Notícia:
    {news}

    Resposta anterior:
    {previous_response}
    """
).strip()
