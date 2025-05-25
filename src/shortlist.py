import pandas as pd

def shortlist_applications(
    df: pd.DataFrame,
    k: int = None,
    threshold: float = None,
    weight_necessity: float = 0.55,
    weight_length: float = 0.1,
    weight_usage: float = 0.35
) -> pd.DataFrame:
    """
    Automatically shortlist grant applications by combining necessity index,
    application length (favoring longer submissions), and the specificity of the
    requested usage list.

    Args:
        df: Processed DataFrame including columns 'necessity_index', 'word_count', and 'Usage'.
        k: Number of top applications to select. Mutually exclusive with threshold.
        threshold: Score threshold above which to select applications. Mutually exclusive with k.
        weight_necessity: Weight for necessity_index (0 to 1).
        weight_length: Weight for length score (0 to 1).
        weight_usage: Weight for usage specificity (0 to 1).

    Returns:
        DataFrame of shortlisted applications sorted by descending combined score.
    """
    # Ensure exactly one of k or threshold is provided
    if (k is None and threshold is None) or (k is not None and threshold is not None):
        raise ValueError("Provide exactly one of k or threshold")

    # Normalize necessity_index (assumed already between 0 and 1)
    necessity = df['necessity_index']

    # Compute length score: longer applications score higher (more context is valued)
    word_counts = df['word_count']
    min_wc, max_wc = word_counts.min(), word_counts.max()
    if max_wc != min_wc:
        length_score = (word_counts - min_wc) / (max_wc - min_wc)
    else:
        length_score = pd.Series([0.5] * len(df), index=df.index)

    # Compute usage score based on *how many* concrete usage items were extracted
    # (previously this was a simple binary flag).  Longer lists are taken as a
    # signal of greater specificity → higher score.  We first count the number
    # of non‑empty items, then min‑max normalise the counts so the resulting
    # score is between 0 and 1 (mirroring the approach used for
    # `length_score`).

    def count_valid_usage(items):
        """Return the number of meaningful usage entries in *items*.

        The Usage column is expected to contain a list of strings (output of
        `extract_usage.extract_usage`). We treat empty strings and the literal
        "None" (case‑insensitive) as non‑entries.
        """
        if not isinstance(items, (list, tuple, set)):
            return 0
        return sum(
            1
            for item in items
            if isinstance(item, str) and item.strip() and item.strip().lower() != "none"
        )

    usage_counts = df["usage"].apply(count_valid_usage)

    min_uc, max_uc = usage_counts.min(), usage_counts.max()
    if max_uc != min_uc:
        usage_score = (usage_counts - min_uc) / (max_uc - min_uc)
    else:
        # If all rows have identical counts (e.g. all zero), assign a neutral 0.5
        usage_score = pd.Series([0.5] * len(df), index=df.index)

    # Combine scores with normalized weights
    total_weight = weight_necessity + weight_length + weight_usage
    weights = {
        'necessity': weight_necessity / total_weight,
        'length': weight_length / total_weight,
        'usage': weight_usage / total_weight,
    }
    combined = (
        weights['necessity'] * necessity +
        weights['length'] * length_score +
        weights['usage'] * usage_score
    )
    df = df.copy()
    df['shortlist_score'] = combined

    # Select applications based on k or threshold
    df_sorted = df.sort_values('shortlist_score', ascending=False)
    if k is not None:
        result = df_sorted.head(k)
    else:
        result = df_sorted[df_sorted['shortlist_score'] >= threshold]

    return result
