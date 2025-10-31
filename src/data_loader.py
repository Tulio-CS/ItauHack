"""Utilitários para carregar bases de notícias sem dependências externas."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, Iterator, List


class UnsupportedFormatError(RuntimeError):
    """Erro disparado quando o formato de arquivo não é suportado."""


def _read_csv(path: Path) -> Iterator[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield {k: (v or "") for k, v in row.items()}


def _read_parquet_placeholder(path: Path) -> Iterator[Dict[str, str]]:
    """Placeholder que instrui sobre a falta de suporte a Parquet.

    O ambiente isolado não fornece bibliotecas como pandas/pyarrow. Para
    manter o fluxo funcional, indicamos um erro claro para que o usuário
    possa converter os arquivos previamente ou instalar as dependências
    necessárias ao executar o script fora do contêiner.
    """

    raise UnsupportedFormatError(
        "Leitura de arquivos Parquet indisponível neste ambiente."
        " Converta previamente para CSV/JSON ou execute o script com"
        " dependências como pandas/pyarrow instaladas."
    )


READERS = {
    ".csv": _read_csv,
    ".parquet": _read_parquet_placeholder,
    ".pq": _read_parquet_placeholder,
}


def load_records(paths: Iterable[Path]) -> List[Dict[str, str]]:
    """Carrega registros agregando todos os arquivos informados."""

    records: List[Dict[str, str]] = []
    for path in paths:
        suffix = path.suffix.lower()
        reader = READERS.get(suffix)
        if reader is None:
            raise UnsupportedFormatError(f"Formato não suportado: {path}")
        records.extend(reader(path))
    return records

