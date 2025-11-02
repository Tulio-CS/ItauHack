"""Rotinas mínimas para criação de gráficos PNG sem dependências externas."""

from __future__ import annotations

import math
import struct
import zlib
from pathlib import Path
from typing import Iterable, List


def _create_canvas(width: int, height: int, color: tuple[int, int, int]) -> List[int]:
    r, g, b = color
    return [channel for _ in range(width * height) for channel in (r, g, b)]


def _set_pixel(pixels: List[int], width: int, height: int, x: int, y: int, color: tuple[int, int, int]) -> None:
    if x < 0 or y < 0 or x >= width or y >= height:
        return
    index = (y * width + x) * 3
    pixels[index : index + 3] = list(color)


def _draw_line(
    pixels: List[int],
    width: int,
    height: int,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    color: tuple[int, int, int],
) -> None:
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    while True:
        _set_pixel(pixels, width, height, x0, y0, color)
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x0 += sx
        if e2 <= dx:
            err += dx
            y0 += sy


def _encode_png(width: int, height: int, pixels: List[int]) -> bytes:
    raw_rows = []
    for y in range(height):
        row_start = y * width * 3
        row_data = bytes([0]) + bytes(pixels[row_start : row_start + width * 3])
        raw_rows.append(row_data)
    raw_data = b"".join(raw_rows)
    compressor = zlib.compressobj()
    compressed = compressor.compress(raw_data) + compressor.flush()

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    header = chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
    body = chunk(b"IDAT", compressed)
    end = chunk(b"IEND", b"")
    return b"\x89PNG\r\n\x1a\n" + header + body + end


def save_line_plot(
    path: Path,
    values: Iterable[float],
    width: int = 900,
    height: int = 480,
    margin: int = 48,
    background: tuple[int, int, int] = (255, 255, 255),
    axis_color: tuple[int, int, int] = (120, 120, 120),
    line_color: tuple[int, int, int] = (8, 81, 156),
) -> None:
    values = list(values)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not values:
        pixels = _create_canvas(width, height, background)
        data = _encode_png(width, height, pixels)
        path.write_bytes(data)
        return

    pixels = _create_canvas(width, height, background)
    min_val = min(values)
    max_val = max(values)
    if math.isclose(min_val, max_val):
        max_val = min_val + 1.0

    plot_width = width - 2 * margin
    plot_height = height - 2 * margin

    # Eixos
    _draw_line(pixels, width, height, margin, height - margin, width - margin, height - margin, axis_color)
    _draw_line(pixels, width, height, margin, margin, margin, height - margin, axis_color)

    def scale_point(index: int, value: float) -> tuple[int, int]:
        x = margin + int(round(index * plot_width / max(len(values) - 1, 1)))
        normalized = (value - min_val) / (max_val - min_val)
        y = height - margin - int(round(normalized * plot_height))
        return x, y

    previous = None
    for idx, value in enumerate(values):
        x, y = scale_point(idx, value)
        if previous is not None:
            _draw_line(pixels, width, height, previous[0], previous[1], x, y, line_color)
        previous = (x, y)

    data = _encode_png(width, height, pixels)
    path.write_bytes(data)
