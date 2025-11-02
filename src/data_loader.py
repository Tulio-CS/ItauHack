"""Utilitários para carregar bases de notícias sem dependências externas.

O módulo agora oferece suporte opcional a arquivos Parquet quando bibliotecas
como :mod:`pyarrow` ou :mod:`pandas` estão disponíveis. Caso contrário, uma
mensagem de erro amigável explica como habilitar o recurso.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List


class UnsupportedFormatError(RuntimeError):
    """Erro disparado quando o formato de arquivo não é suportado."""


def _coerce_value(value: Any) -> str:
    """Normaliza valores para strings compatíveis com o pipeline."""

    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        try:
            return value.decode("utf-8")
        except UnicodeDecodeError:
            return value.decode("utf-8", errors="ignore")
    return str(value)


def _normalize_row(row: Dict[str, Any]) -> Dict[str, str]:
    return {key: _coerce_value(value) for key, value in row.items()}


def _read_csv(path: Path) -> Iterator[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield _normalize_row(row)


def _read_parquet_via_pandas(path: Path) -> Iterator[Dict[str, str]]:
    try:
        import pandas as pd  # type: ignore
    except ImportError as exc:  # pragma: no cover - depende do ambiente
        raise UnsupportedFormatError(
            "Leitura de Parquet requer a instalação de 'pyarrow' ou 'pandas'."
        ) from exc

    try:
        frame = pd.read_parquet(path)
    except (ImportError, ValueError) as exc:  # pragma: no cover - depende do ambiente
        raise UnsupportedFormatError(
            "Pandas precisa de um engine Parquet (pyarrow ou fastparquet)."
            " Instale 'pyarrow' para habilitar a leitura."
        ) from exc

    for row in frame.to_dict(orient="records"):
        yield _normalize_row(row)


def _read_parquet(path: Path) -> Iterator[Dict[str, str]]:
    try:
        import pyarrow.parquet as pq  # type: ignore
    except ImportError:  # pragma: no cover - depende do ambiente
        yield from _read_parquet_via_pandas(path)
        return

    table = pq.read_table(path)
    for row in table.to_pylist():
        yield _normalize_row(row)


READERS = {
    ".csv": _read_csv,
    ".parquet": _read_parquet,
    ".pq": _read_parquet,
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

