# src/pr_opss.py

import json
import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from dotenv import load_dotenv
load_dotenv()

from src.prompts import PROMPTS

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# =============== PYDANTIC MODELS ============

class ClassifiedApplication(BaseModel):
    app_id: str = Field(..., description = "The ID of the application")
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
    .sample(10)
    .rename(columns={id_col: "id", freeform_col: "text"})
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
        logger.info(f"Successfuly classified {len(response.applications)} candidates")
        return response

    except Exception as e:
        logger.error(f"Error with LLM classification for PR opportunity analysis: {str(e)}")
        raise

        
# =================== USAGE =================

from src.column_detection import detect_freeform_col, detect_id_col, detect_school_type_col

def main():

    df = pd.read_csv('./data/raw/old_application_format/april-data.csv')

    df.columns = df.columns.str.lower().str.replace(' ', '_')

    freeform_col = detect_freeform_col(df)
    id_col = detect_id_col(df)

    classified_apps_obj = analyse(df, freeform_col, id_col)

    print("AGENT RESPONSE \n")
    print(classified_apps_obj.model_dump_json(indent=2))
    print(type(classified_apps_obj))
    print(len(classified_apps_obj.applications))


if __name__ == '__main__':
    main()


