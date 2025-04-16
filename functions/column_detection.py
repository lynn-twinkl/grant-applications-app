import pandas as pd
import numpy as np
import re
import string

def detect_freeform_answer_col(df, penalty_for_low_uniqueness=0.4):
    """
    Detect the 'freeform_answer' column using heuristics: average length, punctuation, uniqueness.
    Returns the most likely column name or None.
    """
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    if not text_cols:
        return None

    scores = {}
    for col in text_cols:
        series = df[col].dropna().astype(str)
        if series.empty:
            continue
        avg_len = series.apply(len).mean()
        punct_counts = series.apply(lambda x: sum(1 for char in x if char in string.punctuation))
        avg_punct = punct_counts.mean()
        total = len(series)
        unique_ratio = series.nunique() / total if total else 0

        # Weighted composite
        weight_length = 0.4
        weight_punct  = 0.3
        weight_unique = 0.3
        norm_factor = 1e-9  # avoid dividing by 0
        scores[col] = {
            'avg_len': avg_len,
            'avg_punct': avg_punct,
            'unique_ratio': unique_ratio,
        }

    if not scores:
        return None

    # Normalizing across all columns
    max_len   = max(s['avg_len']   for s in scores.values()) or 1e-9
    max_punct = max(s['avg_punct'] for s in scores.values()) or 1e-9

    composite = {}
    for col, s in scores.items():
        norm_len   = s['avg_len'] / max_len
        norm_punct = s['avg_punct'] / max_punct
        comp_score = (0.4 * norm_len) + (0.3 * norm_punct) + (0.3 * s['unique_ratio'])

        # Bonus/penalty for column names
        if "additional_comment" in col.lower():
            comp_score *= 3.1
        if "usage_reason" in col.lower():
            comp_score *= 0.5
        
        # Penalize low uniqueness
        if s['unique_ratio'] < penalty_for_low_uniqueness:
            comp_score *= 0.5

        composite[col] = comp_score
    
    return max(composite, key=composite.get)
