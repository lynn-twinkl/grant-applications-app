import re
import pandas as pd
from tabulate import tabulate

def find_book_candidates(df: pd.DataFrame, freeform_column: str, school_type_column: str) -> pd.Series:

    if freeform_column not in df.columns:
        raise KeyError(f"Column '{freeform_column}' not found in DataFrame.")
    if school_type_column not in df.columns:
        raise KeyError(f"Column '{school_type_column}' not found in DataFrame.")

    # ---------- 1. CHECK INTENT TO ACQUIRE BOOKS -----------

    freeform_text_series = df[freeform_column].astype(str)

    pattern_intent_to_buy = re.compile(r"""
        \b(want|need|request|require|looking\sfor|look\sfor|purchase|buy|seeking|ordering|order|fund|funding)\b # The "intent" words
        (?:                                     # Non-capturing group for words in between
            \s+                                 # One or more spaces
            \w+                                 # One or more word characters
        ){0,5}                                  # Allow 0 to 5 words in between
        \s+                                     # At least one space before the item
        \b(books?|readers?|texts?|reading\smaterial|sets\sof\sbooks)\b   # The "item" words
    """, re.VERBOSE | re.IGNORECASE)

    wants_books = freeform_text_series.str.contains(pattern_intent_to_buy, na=False)

    # ----------- 2. CHECK SCHOOL TYPES --------
    allowed_school_types = [
        'Primary',
        'Nursery',
        'SEND',
        'Special Primary',
        'Childminder',
        'All-Through',
        'Independent Primary'
    ]

    is_allowed_school_type = df[school_type_column].astype(str).isin(allowed_school_types)

    return wants_books & is_allowed_school_type


# ========== USAGE EXAMPLE ==========

def main():
    
    df = pd.read_csv('data/raw/new-application-format-data.csv')

    df['book_candidates'] = find_book_candidates(df, 'Application Info', 'School Type')

    print(df.iloc[:,[0,1,2,-1]].sample(8).to_markdown(tablefmt='grid'))
    print()
    print(f"📚 TOTAL BOOK CANDIDATES = {len(df[df['book_candidates']])}")
    sample = df[df['book_candidates']].sample(1)

    print('\n' + "📄 SAMPLE CANDIDATE" + '\n')
    print(f"(ID: {sample['App ID'].item()}) {sample['Application Info'].item()}")


if __name__ == "__main__":
    main()
