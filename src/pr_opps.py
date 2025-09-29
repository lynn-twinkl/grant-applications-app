# src/pr_opss.py

import json
import pandas as pd
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List, Dict, Any
from dotenv import load_dotenv
load_dotenv()

from src.prompts import PROMPTS

# ================= HELPERS ===============

def format_application_texts(
        shortlisted_apps: pd.DataFrame,
        freeform_col: str,
        id_col: str
        ) -> Dict[str, Any]:
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


def call_llm(applications_text: str):
    """
    Queries LLM with structured output
    """

    class ClassifiedApplication(BaseModel):
        app_id: str = Field(..., description = "The ID of the application")
        fit_for_pr: bool = Field(..., description = "Whether the application is a viable candidate for PR opportunities or not")
        reasoning: str = Field(..., description = "The reasoning for your classification")
        near_sheffield: bool = Field(..., description = "Whether the school is within 50-60 miles of Sheffield, UK")

    class ClassifiedList(BaseModel):
        applications: list[ClassifiedApplication] = Field(..., description = "A list of all classified applications")



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

def classify_applications(shortlisted_apps: pd.DataFrame):

    logging.info("Starting PR opportunity analysis for {len(shortlisted_apps)} candidates")

    try:
        application_texts = format_application_texts(shortlisted_apps)

    except Exception as e:
        logging.error(f"Error formatting applications texts for PR opportunity extraction: {str(e)}")
        raise

    try:
        response = call_llm(application_texts)
        classified_app_obj = response.applications # Extract list of ClassifiedApplication objs

        logging.info("Successfuly classified {len(classified_apps_obj)} candidates")
        return classified_apps_obj

    except Exception as e:
        logging(f"Error with LLM classification for PR opportunity analysis: {str(e)}")
        raise

        
# =================== USAGE =================

from src.column_detection import detect_freeform_col, detect_id_col, detect_school_type_col

def main():

    df = pd.read_csv('./data/raw/old_application_format/april-data.csv')

    df.columns = df.columns.str.lower().str.replace(' ', '_')

    freeform_col = detect_freeform_col(df)
    id_col = detect_id_col(df)

    application_texts = format_application_texts(df, freeform_col, id_col)

    print("📄 RAW JSON STRING")
    print(application_texts)
    print()

    response = call_llm(application_texts)
    classified_apps_obj = response.applications 
    print("AGENT RESPONSE \n")
    print(response.model_dump_json(indent=2))
    print(type(classified_apps_obj))
    print(len(classified_apps_obj))


if __name__ == '__main__':
    main()


