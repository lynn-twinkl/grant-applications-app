################################
# CONFIGURATION
################################

import streamlit as st
import pandas as pd
from io import BytesIO

# -- FUNCTIONS --

from functions.extract_usage import extract_usage
from functions.necessity_index import compute_necessity 
from functions.column_detection import detect_freeform_answer_col
import typing

# -- CACHEABLE PROCESSING --
@st.cache_data
def load_and_process(raw_csv: bytes) -> typing.Tuple[pd.DataFrame, str]:
    """
    Load CSV from raw bytes, detect freeform column, compute necessity scores,
    and extract usage items. Returns processed DataFrame and freeform column name.
    """
    # Read original data
    df_orig = pd.read_csv(BytesIO(raw_csv))
    # Detect narrative column
    freeform_col = detect_freeform_answer_col(df_orig)
    # Compute necessity scores
    scored = df_orig.join(df_orig[freeform_col].apply(compute_necessity))
    # Extract usage via AI
    docs = df_orig[freeform_col].to_list()
    usage = extract_usage(docs)
    scored['Usage'] = usage
    return scored, freeform_col

################################
# APP SCRIPT
################################

st.title("Grant Applications Helper")

uploaded_file = st.file_uploader("Upload grant applications file for analysis", type='csv')

if uploaded_file is not None:
    # Read raw bytes for caching and repeated use
    raw = uploaded_file.read()

    # --- Original Data Preview ---
    st.markdown("""
    ### Data Preview
    Here's the data you uploaded!
    """
    )
    df_orig = pd.read_csv(BytesIO(raw))
    st.dataframe(df_orig)

    # --- Processed Data (cached): add scores & extracted usage ---
    df, freeform_col = load_and_process(raw)

    # -- Interactive Filtering & Review Interface --
    st.sidebar.header("Filters")
    # Filter by necessity index
    min_idx = float(df['necessity_index'].min())
    max_idx = float(df['necessity_index'].max())
    filter_range = st.sidebar.slider(
        "Necessity Index Range", min_value=min_idx, max_value=max_idx, value=(min_idx, max_idx)
    )
    filtered_df = df[df['necessity_index'].between(filter_range[0], filter_range[1])]

    # Sidebar summary
    st.sidebar.markdown(f"**Total Applications:** {len(df)}")
    st.sidebar.markdown(f"**Filtered Applications:** {len(filtered_df)}")

    # Distribution chart
    st.subheader("Necessity Index Distribution")
    st.bar_chart(df['necessity_index'])

    # Review applications
    st.subheader("Applications")
    for idx, row in filtered_df.iterrows():
        with st.expander(f"Application {idx} | Necessity: {row['necessity_index']:.1f}"):
            col1, col2 = st.columns((1, 3))
            col1.metric("Necessity", f"{row['necessity_index']:.1f}")
            col1.metric("Urgency", f"{row['urgency_score']}")
            col1.metric("Severity", f"{row['severity_score']}")
            col1.metric("Vulnerability", f"{row['vulnerability_score']}")
            # Clean usage items
            usage_items = [item for item in row['Usage'] if item and item.lower() != 'none']
            if usage_items:
                col2.markdown("**Extracted Usage Items:**")
                col2.write(", ".join(usage_items))
            else:
                col2.markdown("*No specific usage items extracted.*")
            col2.markdown("**Excerpt:**")
            col2.write(row[freeform_col])
            # Shortlist checkbox
            st.checkbox(
                "Shortlist this application",
                key=f"shortlist_{idx}"
            )

    # Shortlist summary and download
    shortlisted = [
        i for i in filtered_df.index
        if st.session_state.get(f"shortlist_{i}", False)
    ]
    st.sidebar.markdown(f"**Shortlisted:** {len(shortlisted)}")
    if shortlisted:
        csv = df.loc[shortlisted].to_csv(index=False).encode('utf-8')
        st.sidebar.download_button(
            "Download Shortlist", csv, "shortlist.csv", "text/csv"
        )

