"""Tracking scoreboard column contract.

Human Markdown is rendered by ``scoreboard_format.render_markdown``
(Quality + Counts). This list is for CSV/JSON completeness, not the MD layout.
"""

from __future__ import annotations

SCOREBOARD_COLUMNS = [
    "class",
    "HOTA",
    "DetA",
    "AssA",
    "LocA",
    "DetRe",
    "DetPr",
    "sMOTSA",
    "IDF1",
    "IDSW",
    "Pred",
    "GT",
    "IDs",
    "GT_IDs",
]

SCOREBOARD_META_KEYS = [
    "benchmark",
    "split",
    "tracker",
    "model",
    "conf",
    "prompts",
]

METRIC_DEFS = {
    "HOTA": "Detection + association (main number).",
    "DetA": "Detection accuracy.",
    "AssA": "Association / ID stability.",
    "LocA": "Mask IoU of matched pairs.",
    "DetRe": "Detection recall.",
    "DetPr": "Detection precision (↓ ⇒ extra FPs).",
    "sMOTSA": "Soft MOTS accuracy (TrackEval sMOTA).",
    "IDF1": "Identity F1.",
    "IDSW": "Identity switches (↓ better).",
    "Pred": "Predicted detections (mask instances).",
    "GT": "Ground-truth detections.",
    "IDs": "Predicted unique track IDs.",
    "GT_IDs": "Ground-truth unique track IDs.",
}

BENCHMARKS = {
    "kitti_mots": {
        "classes": ["car", "pedestrian"],
        "gt_kind": "mask",
        "trackeval_dataset": "KittiMOTS",
        "official_splits": ["train", "val"],
        "note": "Driving domain. Primary ADAS benchmark.",
    },
    "mots_challenge": {
        "classes": ["pedestrian"],
        "gt_kind": "mask",
        "trackeval_dataset": "MOTSChallenge",
        "official_splits": ["train"],
        "note": "Crowded pedestrians (MOTS20). Stresses AssA / IDSW.",
    },
    "bdd_mots": {
        "classes": ["person", "bicycle", "car", "motorcycle", "bus", "truck"],
        "gt_kind": "mask",
        "trackeval_dataset": "RobMOTS",
        "official_splits": ["train"],
        "note": (
            "BDD driving via RobMOTS. Eval classes only (clsmap). "
            "No traffic-sign / traffic-light GT — those are qualitative-only."
        ),
    },
}

CONCEPT_TO_CLASS = {
    "car": 1,
    "vehicle": 1,
    "truck": 1,
    "person": 2,
    "pedestrian": 2,
    "people": 2,
}

CONCEPT_TO_CLASS_ROBMOTS = {
    "person": 1,
    "pedestrian": 1,
    "people": 1,
    "bicycle": 2,
    "bike": 2,
    "car": 3,
    "vehicle": 3,
    "motorcycle": 4,
    "motorbike": 4,
    "bus": 6,
    "truck": 8,
}
