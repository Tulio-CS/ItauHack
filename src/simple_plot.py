"""Minimal SVG line chart renderer without external dependencies."""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Sequence, Tuple

Point = Tuple[float, float]


def _ensure_range(min_value: float, max_value: float) -> Tuple[float, float]:
    if math.isclose(min_value, max_value):
        epsilon = 1.0 if math.isclose(min_value, 0.0) else abs(min_value) * 0.01
        return min_value - epsilon, max_value + epsilon
    return min_value, max_value


def _grid_lines(
    axis_start: Tuple[float, float],
    axis_end: Tuple[float, float],
    steps: int,
    orientation: str,
) -> List[str]:
    lines: List[str] = []
    if steps <= 0:
        return lines

    x0, y0 = axis_start
    x1, y1 = axis_end
    for step in range(1, steps):
        ratio = step / steps
        if orientation == "horizontal":
            pos = y0 + ratio * (y1 - y0)
            lines.append(
                (
                    f'<line x1="{x0:.2f}" y1="{pos:.2f}" '
                    f'x2="{x1:.2f}" y2="{pos:.2f}" '
                    'stroke="#d0d0d0" stroke-width="1" stroke-dasharray="4 4" />'
                )
            )
        else:
            pos = x0 + ratio * (x1 - x0)
            lines.append(
                (
                    f'<line x1="{pos:.2f}" y1="{y0:.2f}" '
                    f'x2="{pos:.2f}" y2="{y1:.2f}" '
                    'stroke="#d0d0d0" stroke-width="1" stroke-dasharray="4 4" />'
                )
            )
    return lines


def save_line_chart(
    path: Path,
    points: Sequence[Point],
    title: str = "",
    y_label: str = "",
    color: str = "#2266cc",
) -> None:
    """Render a simple SVG line chart for the provided points."""

    if not points:
        raise ValueError("Nenhum ponto fornecido para o gráfico")

    width, height = 800.0, 400.0
    padding_left, padding_right = 60.0, 20.0
    padding_top, padding_bottom = 50.0, 40.0

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    x_min, x_max = _ensure_range(min(xs), max(xs))
    y_min, y_max = _ensure_range(min(ys), max(ys))

    def project(point: Point) -> Tuple[float, float]:
        x_norm = (point[0] - x_min) / (x_max - x_min)
        y_norm = (point[1] - y_min) / (y_max - y_min)
        x_px = padding_left + x_norm * (width - padding_left - padding_right)
        y_px = height - padding_bottom - y_norm * (height - padding_top - padding_bottom)
        return x_px, y_px

    projected = [project(point) for point in points]

    axis_x0, axis_y0 = padding_left, height - padding_bottom
    axis_x1, axis_y1 = width - padding_right, padding_top

    svg_elements: List[str] = [
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff" />',
        f'<line x1="{axis_x0}" y1="{axis_y0}" x2="{axis_x1}" y2="{axis_y0}" stroke="#000000" stroke-width="2" />',
        f'<line x1="{axis_x0}" y1="{axis_y0}" x2="{axis_x0}" y2="{axis_y1}" stroke="#000000" stroke-width="2" />',
    ]

    svg_elements.extend(
        _grid_lines((axis_x0, axis_y0), (axis_x1, axis_y1), 5, orientation="horizontal")
    )

    svg_elements.append(
        f'<polyline fill="none" stroke="{color}" stroke-width="2" points="'
        + " ".join(f"{x:.2f},{y:.2f}" for x, y in projected)
        + '" />'
    )

    if title:
        svg_elements.append(
            f'<text x="{width / 2}" y="{padding_top / 2}" '
            'text-anchor="middle" font-family="Arial" font-size="20">'
            f"{title}</text>"
        )

    if y_label:
        svg_elements.append(
            f'<text x="{padding_left / 2}" y="{(height - padding_bottom + padding_top) / 2}" '
            'text-anchor="middle" font-family="Arial" font-size="16" transform="rotate(-90 '
            f'{padding_left / 2},{(height - padding_bottom + padding_top) / 2})">{y_label}</text>'
        )

    # tick labels along x axis
    for idx, (x_value, (x_px, _)) in enumerate(zip(xs, projected)):
        if idx % max(1, len(xs) // 10) == 0:
            svg_elements.append(
                f'<text x="{x_px:.2f}" y="{axis_y0 + 20:.2f}" text-anchor="middle" '
                'font-family="Arial" font-size="12">'
                f"{x_value:.0f}</text>"
            )

    # tick labels along y axis
    for step in range(6):
        ratio = step / 5 if step else 0.0
        y_value = y_min + ratio * (y_max - y_min)
        y_px = axis_y0 - ratio * (axis_y0 - axis_y1)
        svg_elements.append(
            f'<text x="{axis_x0 - 10:.2f}" y="{y_px + 4:.2f}" text-anchor="end" '
            'font-family="Arial" font-size="12">'
            f"{y_value:.2f}</text>"
        )

    svg_content = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{int(width)}" height="{int(height)}" '
        f'viewBox="0 0 {int(width)} {int(height)}">\n'
        + "\n".join(svg_elements)
        + "\n</svg>\n"
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg_content, encoding="utf-8")
