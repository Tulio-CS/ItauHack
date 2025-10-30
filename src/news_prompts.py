"""Prompt templates used across the news analysis pipeline."""

from __future__ import annotations

from textwrap import dedent


RELEVANCE_PROMPT = dedent(
    """
    You are a meticulous market analyst. Classify the headline below using one
    of the following categories: `market_moving`, `fluff_marketing`, or
    `irrelevant`.

    Guidelines:
    - `market_moving`: events that can influence profits, production, layoffs,
      mergers, lawsuits, new guidance, or any fact that affects the financial or
      operational performance of the company.
    - `fluff_marketing`: promotional campaigns, institutional pieces, awards,
      social initiatives, or generic interviews.
    - `irrelevant`: headlines unrelated to the company performance.

    Respond **only** with a valid JSON object using double quotes. Examples of
    valid answers:
    {{"relevance": "market_moving"}}
    {{"relevance": "fluff_marketing"}}
    {{"relevance": "irrelevant"}}

    Headline:
    {news}
    """
).strip()


RELEVANCE_RETRY_PROMPT = dedent(
    """
    The previous answer was invalid. Produce EXACTLY ONE valid JSON object in
    the format:
    {{"relevance": "market_moving"}}
    The value must be `market_moving`, `fluff_marketing`, or `irrelevant`.
    Do not add explanations, line breaks, or additional text.

    Headline:
    {news}

    Previous answer:
    {previous_response}
    """
).strip()


STRUCTURED_EVENT_PROMPT = dedent(
    """
    You are a financial analyst. Extract the key facts from the headline below
    and output a JSON object that matches the schema (all keys are required):

    {{
      "evento_tipo": string summarising the main event,
      "sentimento_geral": "positivo" | "negativo" | "neutro" | "misto",
      "impacto": {{
        "nota": integer from 1 to 10 describing potential price impact,
        "justificativa": short textual justification (max 40 words)
      }},
      "metricas": [
        {{
          "metrica": metric name (e.g. receita, EPS, guidance, produção),
          "valor": number or null when not provided,
          "expectativa": number or null when mentioned,
          "resultado": "beat" | "miss" | "inline" | "desconhecido"
        }}
      ]
    }}

    Think through the reasoning privately and answer **ONLY** with the final
    JSON object, using double quotes.

    Headline:
    {news}
    """
).strip()


STRUCTURED_EVENT_RETRY_PROMPT = dedent(
    """
    The previous answer was invalid. Regenerate ONE valid JSON object following
    the exact schema below (all keys are mandatory and must use double quotes):

    {{
      "evento_tipo": string,
      "sentimento_geral": "positivo" | "negativo" | "neutro" | "misto",
      "impacto": {{"nota": integer 1-10, "justificativa": string}},
      "metricas": [
        {{
          "metrica": string,
          "valor": number or null,
          "expectativa": number or null,
          "resultado": "beat" | "miss" | "inline" | "desconhecido"
        }}
      ]
    }}

    Output only the JSON object. Do not add commentary.

    Headline:
    {news}

    Previous answer:
    {previous_response}
    """
).strip()
