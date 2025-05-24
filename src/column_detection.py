"""
column_detect.py  ──  tiny heuristics for finding ID and free‑text columns
"""
from __future__ import annotations  # harmless on 3.11+, useful on 3.7‑3.10
import re
import string
from typing import Sequence, Dict, Tuple, Optional

import pandas as pd


# ========= HELPER FUNCTIONS ========

def _max_or_eps(values, eps: float = 1e-9) -> float:
    """Avoid divide‑by‑zero during normalisation."""
    return max(values) or eps


def _normalise(value: float, max_value: float) -> float:
    return value / max_value if max_value else 0.0

# ========== DETECT FREEFORM COL FUNCTION ============   

def detect_freeform_col(
    df: pd.DataFrame,
    *,
    length_weight: float = 0.4,
    punct_weight: float = 0.3,
    unique_weight: float = 0.3,
    low_uniqueness_penalty: float = 0.4,
    name_boosts: dict[str, float] | None = None,
    min_score: float = 0.50,
    return_scores: bool = False,
) -> str | None | Tuple[str | None, Dict[str, float]]:
    """
    Guess which *object* column contains free‑text answers or comments.

    A good free‑text column tends to be longish, rich in punctuation,
    and fairly unique row‑to‑row.

    name_boosts
        e.g. ``{"additional_comment": 3.1, "usage_reason": 0.5}``
        Multiplicative factors applied if the token appears in the header.
    """
    name_boosts = name_boosts or {}
    obj_cols = df.select_dtypes(include=["object"]).columns

    # quick exit
    if not obj_cols.size:
        return (None, {}) if return_scores else None

    # pre‑compute raw metrics
    raw: Dict[str, dict[str, float]] = {}
    for col in obj_cols:
        ser = df[col].dropna().astype(str)
        if ser.empty:
            continue
        raw[col] = {
            "avg_len": ser.str.len().mean(),
            "avg_punct": ser.apply(lambda s: sum(c in string.punctuation for c in s)).mean(),
            "unique_ratio": ser.nunique() / len(ser),
        }

    if not raw:
        return (None, {}) if return_scores else None

    # normalisers
    max_len = _max_or_eps([m["avg_len"] for m in raw.values()])
    max_punc = _max_or_eps([m["avg_punct"] for m in raw.values()])

    # composite scores
    scores: Dict[str, float] = {}
    for col, m in raw.items():
        score = (
            length_weight * _normalise(m["avg_len"], max_len)
            + punct_weight * _normalise(m["avg_punct"], max_punc)
            + unique_weight * m["unique_ratio"]
        )

        # header boosts / penalties
        for token, factor in name_boosts.items():
            if token in col.lower():
                score *= factor

        # penalise low uniqueness
        if m["unique_ratio"] < low_uniqueness_penalty:
            score *= 0.5

        scores[col] = score

    best_col, best_score = max(scores.items(), key=lambda kv: kv[1])
    passed = best_score >= min_score

    if return_scores:
        return (best_col if passed else None, scores)
    return best_col if passed else None


# ========= DETECT ID COLUMN =========

def detect_id_col(df: pd.DataFrame) -> str | None:
    n_rows = len(df)

    # 1) Name‐based detection
    name_pattern = re.compile(r'\b(id|identifier|key)\b', re.IGNORECASE)
    for col in df.columns:
        if name_pattern.search(col):
            return col

    # 2) Uniqueness detection: columns where every row is unique
    unique_cols = [
        col for col in df.columns
        if df[col].nunique(dropna=False) == n_rows
    ]
    if not unique_cols:
        return None

    # 3) Prioritise int cols over object cols when both are unique
    non_unnamed = [c for c in unique_cols if not c.startswith("Unnamed")]
    candidates = non_unnamed or unique_cols

    # 4) Prefer integer dtypes among candidates
    for col in candidates:
        if pd.api.types.is_integer_dtype(df[col]):
            return col

    # Fallback: return the first candidate
    return candidates[0]

