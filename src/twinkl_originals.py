import re
import pandas as pd

def find_book_candidates(df: pd.DataFrame, column: str) -> pd.Series:

    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in DataFrame")

    series = df[column].astype(str)

    pattern_books = r'\bbooks?\b'
    pattern_level = r'\b(ks1|ks2|primary|eyfs|early years|nursery)\b'

    has_books = series.str.contains(pattern_books, case=False, na=False)
    is_primary = series.str.contains(pattern_level, case=False, na=False)

    
    return has_books & is_primary


