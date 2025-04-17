################################
# CONFIGURATION
################################

import streamlit as st
import pandas as pd
from io import BytesIO

# ---- FUNCTIONS ----

from functions.extract_usage import extract_usage
from functions.necessity_index import compute_necessity, index_scaler
from functions.column_detection import detect_freeform_answer_col
import typing

# ---- CACHEABLE PROCESSING ----

@st.cache_data
def load_and_process(raw_csv: bytes) -> typing.Tuple[pd.DataFrame, str]:
    """
    Load CSV from raw bytes, detect freeform column, compute necessity scores,
    and extract usage items. Returns processed DataFrame and freeform column name.
    """
    # Read uploaded data 
    df_orig = pd.read_csv(BytesIO(raw_csv))
    # Detect freeform column
    freeform_col = detect_freeform_answer_col(df_orig)
    # Compute necessity scores
    scored = df_orig.join(df_orig[freeform_col].apply(compute_necessity))
    scored['necessity_index'] = index_scaler(scored['necessity_index'].values)
    # LangChain function for extracting usage
    docs = df_orig[freeform_col].to_list()
    usage = extract_usage(docs)
    scored['Usage'] = usage
    return scored, freeform_col

################################
# APP SCRIPT
################################

st.title("Community Collections Helper")

uploaded_file = st.file_uploader("Upload grant applications file for analysis", type='csv')

if uploaded_file is not None:
    # Read raw bytes for caching and repeated use --> this ensure all the processing isn't repeated when a user changes the filters
    raw = uploaded_file.read()

    with st.expander("Data Preview", icon='📊'):
        st.markdown("Here's the data you uploaded")
        df_orig = pd.read_csv(BytesIO(raw))
        st.dataframe(df_orig)

    ## ---- PROCESSED DATA (CACHED) ----

    df, freeform_col = load_and_process(raw)

    ## ---- INTERACTIVE FILTERING & REVIEW INTERFACE ----
    
    st.sidebar.title("Filters")
    min_idx = float(df['necessity_index'].min())
    max_idx = float(df['necessity_index'].max())
    filter_range = st.sidebar.slider(
        "Necessity Index Range", min_value=min_idx, max_value=max_idx, value=(min_idx, max_idx)
    )
    filtered_df = df[df['necessity_index'].between(filter_range[0], filter_range[1])]

    # Sidebar summary
    st.sidebar.markdown(f"**Total Applications:** {len(df)}")
    st.sidebar.markdown(f"**Filtered Applications:** {len(filtered_df)}")

    ## ---- NECESSITY INDEX CHART ----

    st.header("Processed Applications")
    st.markdown("#### Necessity Index Distribution")

    with st.expander("Learn about our indexing algorithm", icon='🌱'):
        st.markdown(
                """
                This algorithm is designed to help us evaluate grant applications by assigning a "necessity score" to each application. Here's how it works:

                #### Lexicon

                **Keywords:** The algorithm looks for specific keywords in the application text. These keywords are grouped into five main categories:
                - **Urgency:** Words like "urgent" and "immediate" that express pressing needs.
                - **Severity:** Words like "severe" and "desperate" that indicate serious issues.
                - **Vulnerability:** Words like "disability" "SEND", "underserved" that represent at-risk groups.
                - **Emotional Appeal:** Words like "hope", "committed", "passionate", and "love" that communicate an emotional connection to the mission of each application.
                - **Superlatives:** Words like "completely", "massive", and "significantly" that can serve as proxies for passionate writing.

                #### Weights

                Each category is **assigned a weight accroding to its importance** with **urgency and vulnerability markers** having the highest weights, followed by **severity markers**, and lastly, **superlatives and emotional appeal markers**.

                #### Scoring Process

                - **Need Identification:** The algorithm scans through the application text to count
                how many times each keyword appears.
                - **Weighted Scoring:** It multiplies these counts by the corresponding category weight to compute a total "necessity index", and then normalises the values to produce a range between 0 and 1.

                #### What It Means

                Together, all these scores help can help us rank the applications based on how critical the prize is to each school, with a **higher necessity index suggesting a more compelling case** and a lower score suggesting a "want" rather than a "need".
                """
                )

    st.bar_chart(df['necessity_index'])

    # Review applications
    st.subheader("Filtered Applications")
    st.markdown("To filter applications, use the app's side panel on the left-hand side.")
    for idx, row in filtered_df.iterrows():
        with st.expander(f"Application \#{idx} | Necessity: {row['necessity_index']:.1f}"):
            col1, col2 = st.columns((1, 3))
            col1.metric("Necessity", f"{row['necessity_index']:.1f}")
            col1.metric("Urgency", f"{row['urgency_score']}")
            col1.metric("Severity", f"{row['severity_score']}")
            col1.metric("Vulnerability", f"{row['vulnerability_score']}")
            # Clean usage items
            usage_items = [item for item in row['Usage'] if item and item.lower() != 'none']
            col2.markdown("**EXCERPT**")
            col2.write(row[freeform_col])
            if usage_items:
                col2.markdown("**EXTRACTED USAGE ITEMS:**")
                # Display usage items as colored pills
                pills_html = "".join(
                    f"<span style='display:inline-block;background-color:#D7E0FB;color:#3C63C9;border-radius:20px;padding:4px 10px;margin:2px;'>{item}</span>"
                    for item in usage_items
                )
                col2.markdown(pills_html, unsafe_allow_html=True)
            else:
                col2.markdown("*No specific usage items extracted.*")
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

