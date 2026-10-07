"""Render MOTS / HOTA scoreboards as readable research Markdown.

Never one wide mega-table. Split Quality vs Counts. No <details> dumps.
"""

from __future__ import annotations

from typing import Any

CLASS_ORDER = {"car": 0, "pedestrian": 1, "person": 2}


def _f1(v: Any) -> str:
    if v is None:
        return "—"
    try:
        return f"{float(v):.1f}"
    except (TypeError, ValueError):
        return str(v)


def _i(v: Any) -> str:
    if v is None:
        return "—"
    try:
        return f"{int(round(float(v))):,}"
    except (TypeError, ValueError):
        return str(v)


def _delta(pred: Any, gt: Any) -> str:
    if pred is None or gt is None:
        return "—"
    try:
        d = int(round(float(pred))) - int(round(float(gt)))
        return f"{d:+,d}"
    except (TypeError, ValueError):
        return "—"


def _sorted_rows(rows: list[dict]) -> list[dict]:
    return sorted(rows, key=lambda r: CLASS_ORDER.get(str(r.get("class")), 99))


def title_line(meta: dict) -> str:
    model = meta.get("model") or "model"
    bench = (meta.get("benchmark") or "benchmark").replace("_", " ")
    split = meta.get("split") or ""
    conf = meta.get("conf")
    conf_s = f", conf={conf}" if conf is not None else ""
    return f"{bench} · {split} · {model}{conf_s}".strip(" ·")


def render_markdown(rows: list[dict], meta: dict | None = None) -> str:
    meta = dict(meta or {})
    ordered = _sorted_rows(rows)
    lines = [
        f"# {title_line(meta)}",
        "",
        "## Quality",
        "",
        "| Class | HOTA | DetA | AssA | LocA | sMOTSA | IDF1 | DetPr |",
        "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in ordered:
        lines.append(
            f"| {r.get('class', '—')} | {_f1(r.get('HOTA'))} | {_f1(r.get('DetA'))} | "
            f"{_f1(r.get('AssA'))} | {_f1(r.get('LocA'))} | {_f1(r.get('sMOTSA'))} | "
            f"{_f1(r.get('IDF1'))} | {_f1(r.get('DetPr'))} |"
        )

    lines.extend(
        [
            "",
            "## Counts",
            "",
            "| Class | Pred | GT | Δdet | IDs | GT IDs | Δid | IDSW |",
            "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for r in ordered:
        lines.append(
            f"| {r.get('class', '—')} | {_i(r.get('Pred'))} | {_i(r.get('GT'))} | "
            f"{_delta(r.get('Pred'), r.get('GT'))} | {_i(r.get('IDs'))} | "
            f"{_i(r.get('GT_IDs'))} | {_delta(r.get('IDs'), r.get('GT_IDs'))} | "
            f"{_i(r.get('IDSW'))} |"
        )
    lines.append("")
    return "\n".join(lines)
