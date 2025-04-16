################################
# CONFIGURATION
################################

import streamlit as st
import pandas as pd

# -- FUNCTIONS --

from functions.extract_usage import extract_usage
from functions.necessity_index import compute_necessity 
from functions.column_detection import detect_freeform_answer_col

################################
# APP SCRIPT
################################

st.title("Grant Applications Helper")

uploaded_file = st.file_uploader("Upload grant applications file for analysis", type='csv')

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.markdown("""
    ### Data Preview
    Here's the data you uploaded!
    """
                )

    st.dataframe(df)

    ## 0. PREPROCESSING

    freeform_col = detect_freeform_answer_col(df) # <- detects the long-form column used for processing

    docs = df[freeform_col].to_list()

    ## 1. EXTRACT GRANT USAGE

    extracted_usage = extract_usage(docs)

