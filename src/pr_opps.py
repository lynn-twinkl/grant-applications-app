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


    records = (
    shortlisted_apps[[id_col, freeform_col, 'postcode']]
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


# ============ PUBLIC ENTRYPOINT ==============

def analyse(
        shortlisted_apps: pd.DataFrame,
        freeform_col: str,
        id_col: str
        ) -> ClassifiedList:

    logger.info(f"Starting PR opportunity analysis for {len(shortlisted_apps)} candidates")

    try:
        application_texts = format_application_texts(shortlisted_apps, freeform_col, id_col)
        logger.info(f"Succesfully stringified {len(shortlisted_apps)} shortlisted apps")

    except Exception as e:
        logger.error(f"Error formatting applications texts for PR opportunity extraction: {str(e)}")
        raise

    try:
        logger.info("Attempting LLM classification...")
        response = call_llm(application_texts)
        # Use `response.model_dump_json(indent=2)` to stringify Pydantic object
        logger.info(f"Successfuly classified {len(response.applications)} candidates for PR opportunities")

    except Exception as e:
        logger.error(f"Error with LLM classification for PR opportunity analysis: {str(e)}")
        raise

    try:
        classified_df = pd.DataFrame(response.model_dump()['applications']) # Pydantic-native conversion
        classified_df = classified_df.merge(
                shortlisted_apps[[freeform_col, id_col]],
                left_on = 'app_id',
                right_on = id_col,
                how = 'left'
        )

        return classified_df

    except Exception as e:
        logger.error(f"Error converting {type(response)} to DataFrame: {str(e)}")
        return None 
        
# =================== USAGE =================

from src.column_detection import detect_freeform_col, detect_id_col, detect_school_type_col
from tabulate import tabulate

def main():

    df = pd.read_csv('./data/raw/head10_sample.csv')

    df.columns = df.columns.str.lower().str.replace(' ', '_')

    freeform_col = detect_freeform_col(df)
    id_col = detect_id_col(df)

    classified_apps_df = analyse(df, freeform_col, id_col)

    print(classified_apps_df.columns)
    print(classified_apps_df.head(2))
    print(type(classified_apps_df))

if __name__ == '__main__':
    main()


