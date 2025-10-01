# src/pr_opss.py

import json
import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from dotenv import load_dotenv
load_dotenv()

from src.prompts import PROMPTS
from src.logger import get_logger

logger = get_logger(__name__)

# =============== PYDANTIC MODELS ============

class ClassifiedApplication(BaseModel):
    app_id: int = Field(..., description = "The ID of the application")
    fit_for_pr: bool = Field(..., description = "Whether the application is a viable candidate for PR opportunities or not")
    reasoning: str = Field(..., description = "The reasoning for your classification")
    near_sheffield: bool = Field(..., description = "Whether the school is within 50-60 miles of Sheffield, UK")

class ClassifiedList(BaseModel):
    applications: list[ClassifiedApplication] = Field(..., description = "A list of all classified applications")


# ================= HELPERS ===============

def format_application_texts(
        shortlisted_apps: pd.DataFrame,
        freeform_col: str,
        id_col: str
        ) -> str:
    """
    Formats dataframe containing shortlist into
    a JSON string to be sent to LLM
    """

    missing = {id_col, freeform_col} - set(shortlisted_apps.columns)
    if missing:
        raise ValueError(f"Missing expected columns: {', '.join(missing)}")

    cols = [id_col, freeform_col]
    if 'postcode' in shortlisted_apps.columns:
        cols.append('postcode')  # include when present

    records = (
        shortlisted_apps[cols]
        .rename(columns={id_col: "app_id", freeform_col: "text"})
        .to_dict(orient="records")
    )

    return json.dumps(records, indent=2, ensure_ascii=False)


def call_llm(applications_text: str) -> ClassifiedList:
    """
    Queries LLM with structured output
    """
    client = OpenAI()
    response = client.responses.parse(
        model="gpt-5-mini",
        input=[
            {"role": "system", "content": PROMPTS['pr_classification_system']},
            {"role": "user", "content": applications_text},
        ],
        text_format=ClassifiedList,
    )
    return response.output_parsed


def _chunk_df(df: pd.DataFrame, size: int):
    """
    Yield successive chunks of a dataframe of length `size`.
    """
    for start in range(0, len(df), size):
        yield df.iloc[start:start + size]


# ============ PUBLIC ENTRYPOINT ==============

def analyse(
        shortlisted_apps: pd.DataFrame,
        freeform_col: str,
        id_col: str,
        batch_size: int = 30
        ) -> pd.DataFrame:
    """
    Analyse shortlisted applications in batches so we send at most `batch_size`
    items to the LLM per request, then concatenate the parsed results.

    Returns a DataFrame with the parsed classifications merged back on to the
    original rows (by `app_id`/`id_col`).
    """

    logger.info(f"Starting PR opportunity analysis for {len(shortlisted_apps)} candidates (batch_size={batch_size})")

    # Defensive lower bound
    if batch_size <= 0:
        raise ValueError("batch_size must be a positive integer")

    all_parsed: list[ClassifiedApplication] = []

    try:
        for i, chunk in enumerate(_chunk_df(shortlisted_apps, batch_size), start=1):
            logger.info(f"[Batch {i}] Preparing {len(chunk)} apps")
            applications_text = format_application_texts(chunk, freeform_col, id_col)
            logger.info(f"[Batch {i}] Sending to LLM...")
            parsed: ClassifiedList = call_llm(applications_text)
            logger.info(f"[Batch {i}] Received {len(parsed.applications)} classifications")
            all_parsed.extend(parsed.applications)

    except Exception as e:
        logger.error(f"Error during batched LLM classification: {str(e)}")
        raise

    if not all_parsed:
        logger.error("No classifications returned from LLM")
        return None

    # Build a single ClassifiedList to keep a consistent path if needed elsewhere
    concatenated = ClassifiedList(applications=all_parsed)

    try:
        classified_df = pd.DataFrame([app.model_dump() for app in concatenated.applications])

        # Merge back original text/id for convenience
        classified_df = classified_df.merge(
            shortlisted_apps[[freeform_col, id_col]],
            left_on='app_id',
            right_on=id_col,
            how='left'
        )

        logger.info(f"Successfully classified {len(classified_df)} candidates in total across batches")
        return classified_df

    except Exception as e:
        logger.error(f"Error converting concatenated results to DataFrame: {str(e)}")
        return None


# =================== USAGE =================

from src.column_detection import detect_freeform_col, detect_id_col, detect_school_type_col
from tabulate import tabulate

def main():
    df = pd.read_csv('./data/raw/head10_sample.csv')
    df.columns = df.columns.str.lower().str.replace(' ', '_')

    freeform_col = detect_freeform_col(df)
    id_col = detect_id_col(df)

    # You can override batch size here if you want
    classified_apps_df = analyse(df, freeform_col, id_col, batch_size=30)

    print(classified_apps_df.columns)
    print(classified_apps_df.head(2))
    print(type(classified_apps_df))

if __name__ == '__main__':
    main()
