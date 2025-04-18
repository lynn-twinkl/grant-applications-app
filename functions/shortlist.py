import pandas as pd

def shortlist_applications(
    df: pd.DataFrame,
    k: int = None,
    threshold: float = None,
    weight_necessity: float = 0.5,
    weight_length: float = 0.3,
    weight_usage: float = 0.2
) -> pd.DataFrame:
    """
    Automatically shortlist grant applications by combining necessity index,
    application length (favoring longer submissions), and whether usage was specified.

    Args:
        df: Processed DataFrame including columns 'necessity_index', 'word_count', and 'Usage'.
        k: Number of top applications to select. Mutually exclusive with threshold.
        threshold: Score threshold above which to select applications. Mutually exclusive with k.
        weight_necessity: Weight for necessity_index (0 to 1).
        weight_length: Weight for length score (0 to 1).
        weight_usage: Weight for usage inclusion (0 to 1).

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

    # Compute usage score: 1 if any usage items specified, else 0
    def has_usage(items):
        return any(
            item and isinstance(item, str) and item.strip().lower() != 'none'
            for item in items
        )

    usage_score = df['Usage'].apply(has_usage).astype(float)

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
    df['auto_shortlist_score'] = combined

    # Select applications based on k or threshold
    df_sorted = df.sort_values('auto_shortlist_score', ascending=False)
    if k is not None:
        result = df_sorted.head(k)
    else:
        result = df_sorted[df_sorted['auto_shortlist_score'] >= threshold]

    return result