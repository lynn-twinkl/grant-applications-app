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

# =================== FREEFORM COL =====================   

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


# ================= ID COLUMN =================

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


# ============== CAREER COLUMN =============

def detect_career_col(
    df: pd.DataFrame,
    *,
    uniqueness_weight: float = 0.5,
    length_weight: float = 0.3,
    punct_weight: float = 0.2,
    name_boosts: dict[str, float] | None = None,
    min_score: float = 0.40,
    high_uniqueness_penalty: float = 0.95,
    return_scores: bool = False,
) -> str | None | Tuple[str | None, Dict[str, float]]:
    """
    Analyzes a DataFrame to find the column that most likely represents a 'career' or 'role'.

    The function operates on heuristics based on common characteristics of a career column:
    1.  **Low Uniqueness**: Values are often repeated (e.g., 'teacher', 'ks1').
    2.  **Short Text**: Entries are typically brief.
    3.  **Minimal Punctuation**: Values are clean strings, not sentences.
    4.  **Header Keywords**: The column name itself is a strong indicator (e.g., 'Career', 'Job').

    Args:
        df: The DataFrame to analyze.
        uniqueness_weight: The importance of having low uniqueness (many repeated values).
        length_weight: The importance of having short text values.
        punct_weight: The importance of having little to no punctuation.
        name_boosts: Multiplicative factors for keyword matches in the column header.
                     Defaults to boosts for 'career', 'job', 'role', and 'position'.
        min_score: The minimum score for a column to be considered a match.
        high_uniqueness_penalty: A uniqueness ratio (e.g., 0.95) above which a column's
                                 score is heavily penalized, as it is unlikely to be
                                 a categorical role column.
        return_scores: If True, returns a tuple containing the best column name and a
                       dictionary of scores for all candidate columns.

    Returns:
        The name of the detected career column, or None if no suitable column is found.
        If return_scores is True, it returns a tuple of (column_name, scores_dict).
    """

    if name_boosts is None:
        name_boosts = {'career': 3.0, 'job': 2.5, 'role': 2.5, 'position': 2.0}

    obj_cols = df.select_dtypes(include=["object"]).columns
    if not obj_cols.size:
        return (None, {}) if return_scores else None

    # Pre-compute raw metrics for each object column
    raw_metrics: Dict[str, dict[str, float]] = {}
    for col in obj_cols:
        # Drop temporary NA's to not skew metrics, then convert to string
        ser = df[col].dropna().astype(str)
        if ser.empty:
            continue
        raw_metrics[col] = {
            "avg_len": ser.str.len().mean(),
            "avg_punct": ser.apply(lambda s: sum(c in string.punctuation for c in s)).mean(),
            "unique_ratio": ser.nunique() / len(ser) if len(ser) > 0 else 0.0,
        }

    if not raw_metrics:
        return (None, {}) if return_scores else None

    # Get max values for normalization across all columns
    max_len = _max_or_eps([m["avg_len"] for m in raw_metrics.values()])
    max_punc = _max_or_eps([m["avg_punct"] for m in raw_metrics.values()])

    # Calculate a final score for each column
    scores: Dict[str, float] = {}
    for col, metrics in raw_metrics.items():
        len_score = 1 - _normalise(metrics["avg_len"], max_len)
        punc_score = 1 - _normalise(metrics["avg_punct"], max_punc)
        uniq_score = 1 - metrics["unique_ratio"]

        score = (
            length_weight * len_score
            + punct_weight * punc_score
            + uniqueness_weight * uniq_score
        )

        # Apply boosts for matching header keywords
        for token, factor in name_boosts.items():
            if token in col.lower().strip():
                score *= factor

        # Apply penalty for columns that are almost entirely unique
        if metrics["unique_ratio"] > high_uniqueness_penalty:
            score *= 0.1  # Heavy penalty

        scores[col] = score

    if not scores:
         return (None, {}) if return_scores else None

    best_col, best_score = max(scores.items(), key=lambda item: item[1])
    passed = best_score >= min_score

    if return_scores:
        return (best_col if passed else None, scores)
    return best_col if passed else None

# =========== USAGE ============

def main():

    df = pd.read_csv('data/raw/new-application-format-data.csv')
    df.columns = df.columns.str.strip()

    print("--- Testing Column Detection Functions ---")

    id_col = detect_id_col(df)
    freeform_col, freeform_scores = detect_freeform_col(df, return_scores=True)
    career_col, career_scores = detect_career_col(df, return_scores=True)

    print(f"\nDetected ID Column: '{id_col}'")
    print(f"Detected Free-Form Column: '{freeform_col}'")
    print(f"Detected Career Column: '{career_col}'")

    print("\n--- Career Column Scores (Higher is better) ---")
    if career_scores:
        sorted_scores = sorted(career_scores.items(), key=lambda item: item[1], reverse=True)
        for col, score in sorted_scores:
            print(f"  - {col:<25}: {score:.4f}")
    else:
        print("No object columns found to score for career.")

if __name__ == '__main__':
    main()
