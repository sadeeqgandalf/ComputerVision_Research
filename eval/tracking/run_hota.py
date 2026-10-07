#!/usr/bin/env python3
"""Run TrackEval HOTA (+ CLEAR, Identity) and write a locked scoreboard table.

Example (after preds exist in TrackEval layout):

  python eval/tracking/run_hota.py \\
    --benchmark kitti_mots --split val \\
    --tracker sam3_1_person_car \\
    --gt-folder outputs/benchmarks/ground_truth/kitti_mots \\
    --trackers-folder outputs/SAM3.1/tracking/predictions/kitti_mots/val
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TRACKEVAL = ROOT / "TrackEval"
SEQMAP_DIR = Path(__file__).resolve().parent / "seqmaps"
TABLES = ROOT / "outputs" / "SAM3.1" / "tracking" / "metrics"

sys.path.insert(0, str(Path(__file__).resolve().parent))
from metrics_spec import METRIC_DEFS, SCOREBOARD_COLUMNS, SCOREBOARD_META_KEYS  # noqa: E402
from scoreboard_format import render_markdown  # noqa: E402


def _mean_or_none(arr):
    if arr is None:
        return None
    try:
        import numpy as np

        a = np.asarray(arr, dtype=float)
        if a.size == 0:
            return None
        return float(np.mean(a))
    except Exception:
        return None


def _scalar(res: dict, key: str):
    if key not in res:
        return None
    val = res[key]
    if hasattr(val, "shape"):
        # HOTA fields are arrays over alpha; report mean over alpha (TrackEval default summary)
        return _mean_or_none(val)
    try:
        return float(val)
    except (TypeError, ValueError):
        return None


def _parse_summary_txt(path: Path) -> dict[str, float]:
    """Parse TrackEval class summary: header line + one COMBINED values line."""
    lines = [ln.strip() for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if len(lines) < 2:
        return {}
    keys = lines[0].split()
    vals = lines[1].split()
    out: dict[str, float] = {}
    for k, v in zip(keys, vals):
        try:
            out[k] = float(v)
        except ValueError:
            continue
    return out


def rows_from_trackeval(output_folder: Path, benchmark: str, split: str, tracker: str) -> list[dict]:
    """Parse TrackEval class summaries into scoreboard rows.

    KittiMOTS writes per-class files like:
      OUTPUT/<benchmark>/<split>/<tracker>/car_summary.txt
      OUTPUT/<benchmark>/<split>/<tracker>/pedestrian_summary.txt
    """
    rows: list[dict] = []
    tracker_dirs = [p for p in Path(output_folder).rglob(tracker) if p.is_dir()]
    if not tracker_dirs:
        tracker_dirs = [output_folder / tracker]

    summary_files: list[Path] = []
    for td in tracker_dirs:
        summary_files.extend(sorted(td.glob("*_summary.txt")))

    if not summary_files:
        rows.append(
            {
                "benchmark": benchmark,
                "split": split,
                "tracker": tracker,
                "class": "PARSE_ERROR",
                "HOTA": None,
                "note": f"no *_summary.txt under {output_folder}",
            }
        )
        return rows

    for path in summary_files:
        cls = path.name.replace("_summary.txt", "")
        m = _parse_summary_txt(path)
        if not m:
            continue
        smotsa = m.get("sMOTA") if m.get("sMOTA") is not None else m.get("sMOTSA")
        rows.append(
            {
                "benchmark": benchmark,
                "split": split,
                "tracker": tracker,
                "class": cls,
                "HOTA": m.get("HOTA"),
                "DetA": m.get("DetA"),
                "AssA": m.get("AssA"),
                "LocA": m.get("LocA"),
                "DetRe": m.get("DetRe"),
                "DetPr": m.get("DetPr"),
                "sMOTSA": smotsa,
                "IDF1": m.get("IDF1"),
                "IDSW": m.get("IDSW"),
                "Pred": m.get("Dets"),
                "GT": m.get("GT_Dets"),
                "IDs": m.get("IDs"),
                "GT_IDs": m.get("GT_IDs"),
                # Kept in JSON for deep debug; omitted from primary MD table.
                "AssRe": m.get("AssRe"),
                "AssPr": m.get("AssPr"),
                "MOTA": m.get("MOTA"),
                "CLR_TP": m.get("CLR_TP"),
                "CLR_FN": m.get("CLR_FN"),
                "CLR_FP": m.get("CLR_FP"),
            }
        )
    return rows


def write_tables(
    rows: list[dict],
    stem: str,
    meta: dict | None = None,
) -> tuple[Path, Path]:
    TABLES.mkdir(parents=True, exist_ok=True)
    json_path = TABLES / f"{stem}.json"
    md_path = TABLES / f"{stem}.md"
    csv_path = TABLES / f"{stem}.csv"

    meta = dict(meta or {})
    if rows:
        for k in SCOREBOARD_META_KEYS:
            meta.setdefault(k, rows[0].get(k))

    payload = {"meta": meta, "rows": rows, "metric_defs": METRIC_DEFS}
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    cols = [c for c in SCOREBOARD_COLUMNS if any(c in r for r in rows)]
    csv_fields = list(SCOREBOARD_META_KEYS) + [
        c
        for c in (
            list(SCOREBOARD_COLUMNS)
            + ["AssRe", "AssPr", "MOTA", "CLR_TP", "CLR_FN", "CLR_FP", "DetRe"]
        )
        if any(c in r for r in rows)
    ]
    # dedupe preserving order
    seen: set[str] = set()
    csv_fields = [x for x in csv_fields if not (x in seen or seen.add(x))]  # type: ignore[func-returns-value]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            row = {k: meta.get(k, r.get(k)) for k in SCOREBOARD_META_KEYS}
            row.update({c: r.get(c) for c in csv_fields if c not in SCOREBOARD_META_KEYS})
            w.writerow(row)

    md_path.write_text(render_markdown(rows, meta), encoding="utf-8")
    return json_path, md_path


def run_eval(
    benchmark: str,
    split: str,
    tracker: str,
    gt_folder: Path,
    trackers_folder: Path,
    output_folder: Path,
    classes: list[str] | None,
    seqmap_file: Path | None = None,
) -> None:
    sys.path.insert(0, str(TRACKEVAL))
    import trackeval  # noqa: WPS433

    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config.update(
        {
            "DISPLAY_LESS_PROGRESS": True,
            "PRINT_CONFIG": False,
            "OUTPUT_SUMMARY": True,
            "OUTPUT_DETAILED": True,
            "PLOT_CURVES": False,
            "PRINT_RESULTS": True,
        }
    )

    if benchmark == "kitti_mots":
        dataset_config = trackeval.datasets.KittiMOTS.get_default_dataset_config()
        dataset_config.update(
            {
                "GT_FOLDER": str(gt_folder),
                "TRACKERS_FOLDER": str(trackers_folder),
                "OUTPUT_FOLDER": str(output_folder),
                "TRACKERS_TO_EVAL": [tracker],
                "CLASSES_TO_EVAL": classes or ["car", "pedestrian"],
                "SPLIT_TO_EVAL": split,
                "PRINT_CONFIG": False,
                "TRACKER_SUB_FOLDER": "data",
                "GT_LOC_FORMAT": "{gt_folder}/instances_txt/{seq}.txt",
                "SEQMAP_FILE": str(SEQMAP_DIR / f"kitti_mots_{split}.seqmap"),
            }
        )
        dataset_list = [trackeval.datasets.KittiMOTS(dataset_config)]
    elif benchmark == "mots_challenge":
        # Prefer SEQ_INFO so we don't require seqinfo.ini / CSV seqmaps.
        seq_info: dict[str, int] = {}
        for line in (SEQMAP_DIR / "mots_challenge_train.seqmap").read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            seq = f"{int(parts[0]):04d}"
            start, end = int(parts[2]), int(parts[3])
            seq_info[seq] = end - start + 1
        dataset_config = trackeval.datasets.MOTSChallenge.get_default_dataset_config()
        dataset_config.update(
            {
                "GT_FOLDER": str(gt_folder),
                "TRACKERS_FOLDER": str(trackers_folder),
                "OUTPUT_FOLDER": str(output_folder),
                "TRACKERS_TO_EVAL": [tracker],
                "CLASSES_TO_EVAL": classes or ["pedestrian"],
                "SPLIT_TO_EVAL": split,
                "PRINT_CONFIG": False,
                "TRACKER_SUB_FOLDER": "data",
                "SKIP_SPLIT_FOL": True,
                "SEQ_INFO": seq_info,
                "GT_LOC_FORMAT": "{gt_folder}/{seq}/gt/gt.txt",
            }
        )
        dataset_list = [trackeval.datasets.MOTSChallenge(dataset_config)]
    elif benchmark == "bdd_mots":
        # RobMOTS layout: GT_FOLDER/train/bdd_mots/{data,seqmap,clsmap}
        # trackers: TRACKERS_FOLDER/train/<tracker>/data/bdd_mots/<seq>.txt
        sm = seqmap_file or (gt_folder / split / "bdd_mots" / "seqmap.txt")
        dataset_config = trackeval.datasets.RobMOTS.get_default_dataset_config()
        dataset_config.update(
            {
                "GT_FOLDER": str(gt_folder),
                "TRACKERS_FOLDER": str(trackers_folder),
                "OUTPUT_FOLDER": str(output_folder),
                "TRACKERS_TO_EVAL": [tracker],
                "SUB_BENCHMARK": "bdd_mots",
                "CLASSES_TO_EVAL": classes
                or ["person", "bicycle", "car", "motorcycle", "bus", "truck"],
                "SPLIT_TO_EVAL": split,
                "PRINT_CONFIG": False,
                "TRACKER_SUB_FOLDER": "data",
                "SEQMAP_FILE": str(sm),
            }
        )
        dataset_list = [trackeval.datasets.RobMOTS(dataset_config)]
    else:
        raise SystemExit(f"Unknown benchmark: {benchmark}")

    metrics_list = [
        trackeval.metrics.HOTA(),
        trackeval.metrics.CLEAR(),
        trackeval.metrics.Identity(),
    ]
    evaluator = trackeval.Evaluator(eval_config)
    evaluator.evaluate(dataset_list, metrics_list)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark",
        choices=["kitti_mots", "mots_challenge", "bdd_mots"],
        required=True,
    )
    parser.add_argument("--split", default="val")
    parser.add_argument("--tracker", required=True)
    parser.add_argument("--gt-folder", type=Path, required=True)
    parser.add_argument("--trackers-folder", type=Path, required=True)
    parser.add_argument(
        "--output-folder",
        type=Path,
        default=ROOT / "outputs" / "SAM3.1" / "tracking" / "results",
    )
    parser.add_argument("--classes", default=None, help="Comma-separated classes")
    parser.add_argument(
        "--seqmap-file",
        type=Path,
        default=None,
        help="Optional RobMOTS seqmap override (smoke / subset eval).",
    )
    parser.add_argument("--table-only", action="store_true", help="Rebuild table from existing results")
    parser.add_argument("--conf", type=float, default=None, help="Conf threshold used at inference (scoreboard meta)")
    parser.add_argument("--model", default="sam3.1", help="Model label for scoreboard meta")
    parser.add_argument("--prompts", default="person,car", help="Prompts for scoreboard meta")
    args = parser.parse_args()

    classes = [c.strip() for c in args.classes.split(",")] if args.classes else None
    out = args.output_folder / args.benchmark / args.split
    out.mkdir(parents=True, exist_ok=True)

    if not args.table_only:
        run_eval(
            benchmark=args.benchmark,
            split=args.split,
            tracker=args.tracker,
            gt_folder=args.gt_folder,
            trackers_folder=args.trackers_folder,
            output_folder=out,
            classes=classes,
            seqmap_file=args.seqmap_file,
        )

    rows = rows_from_trackeval(out, args.benchmark, args.split, args.tracker)
    stem = f"{args.benchmark}_{args.split}_{args.tracker}"
    meta = {
        "benchmark": args.benchmark,
        "split": args.split,
        "tracker": args.tracker,
        "model": args.model,
        "conf": args.conf,
        "prompts": args.prompts,
    }
    json_path, md_path = write_tables(rows, stem, meta=meta)
    print(f"\nScoreboard JSON: {json_path}")
    print(f"Scoreboard MD:   {md_path}")
    for r in rows:
        print(
            f"  {r.get('class')}: HOTA={r.get('HOTA')} DetPr={r.get('DetPr')} "
            f"IDF1={r.get('IDF1')} IDSW={r.get('IDSW')} "
            f"Pred={r.get('Pred')} GT={r.get('GT')} IDs={r.get('IDs')} GT_IDs={r.get('GT_IDs')}"
        )


if __name__ == "__main__":
    main()
