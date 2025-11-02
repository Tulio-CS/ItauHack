"""Pipeline para classificação e extração estruturada de notícias."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _bootstrap_path() -> None:
    """Garante que o pacote ``src`` esteja disponível no ``sys.path``.

    Quando o script é executado diretamente (``python scripts/run_local_llm.py``),
    o diretório do repositório nem sempre é adicionado automaticamente ao
    ``sys.path``. Isso é comum principalmente no Windows, causando o erro
    ``ModuleNotFoundError: No module named 'src'``. Este helper insere o
    diretório raiz do projeto no início da lista de caminhos de importação,
    evitando que o usuário precise configurar ``PYTHONPATH`` manualmente.
    """

    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))


_bootstrap_path()

from src.data_loader import UnsupportedFormatError, load_records
from src.local_llm import LocalLLM


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--files",
        nargs="+",
        type=Path,
        default=[
            Path("data/investing_news.parquet"),
            Path("data/investing_news_nacionais.parquet"),
            Path("data/investing_news_nacionais_que_faltaram.parquet"),
        ],
        help="Lista de arquivos com notícias (CSV/Parquet).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/output/investing_news_structured.jsonl"),
        help="Arquivo de saída no formato JSON Lines.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Processa apenas os primeiros N registros (útil para testes).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Define o nível de log (ex.: DEBUG, INFO, WARNING).",
    )
    return parser.parse_args(argv)


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, str(args.log_level).upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    try:
        records = load_records(args.files)
    except UnsupportedFormatError as exc:  # pragma: no cover - mensagens informativas
        raise SystemExit(str(exc)) from exc

    if args.limit is not None:
        records = records[: args.limit]

    llm = LocalLLM()

    enriched: List[Dict[str, Any]] = []
    for row in records:
        text = " ".join(
            filter(
                None,
                [
                    row.get("headline"),
                    row.get("title"),
                    row.get("content"),
                    row.get("description"),
                ],
            )
        )
        if not text:
            text = row.get("summary") or row.get("text") or ""

        event = llm.classify(text)
        enriched.append(
            {
                "raw": row,
                "structured_event": event.to_dict(),
            }
        )

    ensure_parent(args.output)
    with args.output.open("w", encoding="utf-8") as handle:
        for item in enriched:
            json.dump(item, handle, ensure_ascii=False)
            handle.write("\n")

    print(f"Processados {len(enriched)} registros. Saída em {args.output}.")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

