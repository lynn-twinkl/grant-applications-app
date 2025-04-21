################################
# CONFIGURATION
################################

import streamlit as st
import pandas as pd
import altair as alt
from io import BytesIO
from streamlit_extras.metric_cards import style_metric_cards

# ---- FUNCTIONS ----

from functions.extract_usage import extract_usage
from functions.necessity_index import compute_necessity, index_scaler, qcut_labels
from functions.column_detection import detect_freeform_col
from functions.shortlist import shortlist_applications
from typing import Tuple

##################################
# CACHED PROCESSING FUNCTION
##################################

# -----------------------------------------------------------------------------
# Heavy processing (IO + NLP) is cached to avoid re‑executing when the UI state
# changes. The function only re‑runs if the **file contents** change.
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=True)
def load_and_process(raw_csv: bytes) -> Tuple[pd.DataFrame, str]:
    """
    Load CSV from raw bytes, detect freeform column, compute necessity scores,
    and extract usage items. Returns processed DataFrame and freeform column name.
    """
    # Read Uploaded Data 
    df_orig = pd.read_csv(BytesIO(raw_csv))

    # Detect freeform column
    freeform_col = detect_freeform_col(df_orig)

    #Word Count
    df_orig['word_count'] = df_orig[freeform_col].fillna('').str.split().str.len()

    # Compute Necessity Scores
    scored = df_orig.join(df_orig[freeform_col].apply(compute_necessity))
    scored['necessity_index'] = index_scaler(scored['necessity_index'].values)
    scored['priority'] = qcut_labels(scored['necessity_index'])
    
    # Usage Extraction
    docs = df_orig[freeform_col].to_list()
    scored['Usage'] = extract_usage(docs)

    return scored, freeform_col

# -----------------------------------------------------------------------------
# Derivative computations that rely only on the processed DataFrame are also
# cached. These are lightweight but still benefit from caching because this
# function might be called multiple times during widget interaction.
# -----------------------------------------------------------------------------


@st.cache_data(show_spinner=True)
def compute_shortlist(df: pd.DataFrame) -> pd.DataFrame:
    """Pre‑compute shortlist_score for all rows (used for both modes)."""
    return shortlist_applications(df, k=len(df))

################################
# APP SCRIPT
################################

style_metric_cards(box_shadow=False, border_left_color='#E7F4FF',background_color='#E7F4FF', border_size_px=0, border_radius_px=6)

st.title("Community Collections Helper")

uploaded_file = st.file_uploader("Upload grant applications file for analysis", type='csv')

if uploaded_file is not None:
    # Read file from raw bytes for caching and repeated use --> this ensure all the processing isn't repeated when a user changes the filters
    raw = uploaded_file.read()

    ## ---- PROCESSED DATA (CACHED) ----

    df, freeform_col = load_and_process(raw)

    ## ---- INTERACTIVE FILTERING & REVIEW INTERFACE ----

    with st.sidebar:
        st.title("Shortlist Mode")

        quantile_map = {"strict": 0.75, "generous": 0.5}
        mode = st.segmented_control(
                "Select one option",
                options=["strict", "generous"],
                default="strict",
                )
        
        scored_full = compute_shortlist(df)
        threshold_score = scored_full["shortlist_score"].quantile(quantile_map[mode])
        auto_short_df = scored_full[scored_full["shortlist_score"] >= threshold_score]

        st.title("Filters")
        min_idx = float(df['necessity_index'].min())
        max_idx = float(df['necessity_index'].max())
        filter_range = st.sidebar.slider(
            "Necessity Index Range", min_value=min_idx, max_value=max_idx, value=(min_idx, max_idx)
        )
        filtered_df = df[(~df.index.isin(auto_short_df.index)) & (df['necessity_index'].between(filter_range[0], filter_range[1]))]

        st.markdown(f"**Total Applications:** {len(df)}")
        st.markdown(f"**Filtered Applications:** {len(filtered_df)}")

    ## ----------------- MAIN PANEL ----------------

    tab1, tab2 = st.tabs(["Shortlist Manager","Insights"])

    ## ---------- SHORTLIST MANAGER TAB -----------

    with tab1:
        
        st.header("✨ Automatic Shortlist")
        st.markdown("Here's your **automatically genereated shortlist!** If you'd like to manually add additional applications, you may do so on the section below!")

        csv_auto = auto_short_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Shortlist",
            data=csv_auto,
            file_name="auto_shortlist.csv",
            mime="text/csv",
            icon='⬇️'
        )
        st.markdown("#### Shortlist Preview")
        st.write("")
        total_col, shortlistCounter_col, mode_col = st.columns(3)

        total_col.metric("Applications Submitted", len(df))
        shortlistCounter_col.metric("Shorlist Length",  len(auto_short_df))
        mode_col.metric("Mode", mode)

        shorltist_cols_to_show = [
                'Id',
                freeform_col,
                'Usage',
                'necessity_index',
                'urgency_score',
                'severity_score',
                'vulnerability_score',
                'shortlist_score'
                ]

        st.dataframe(auto_short_df.loc[:, shorltist_cols_to_show], hide_index=True)

        ## REVIEW APPLICATIONS 

        st.header("🌸 Manual Filtering")
        st.markdown(
                """
                Below you'll find applications that **did not** make it into the shortlist for you to manually review or append to the shortlist if desired.

                You may use the **side panel** filters to more easily sort through applications that you'd like to review.
                """
                )

        st.markdown("#### Filtered Applications")
        st.write("")
        for idx, row in filtered_df.iterrows():
            with st.expander(f"Application \#{idx}"):
                st.write("")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Necessity", f"{row['necessity_index']:.1f}")
                col2.metric("Urgency", f"{int(row['urgency_score'])}")
                col3.metric("Severity", f"{int(row['severity_score'])}")
                col4.metric("Vulnerability", f"{int(row['vulnerability_score'])}")

                # HTML for clean usage items 
                usage_items = [item for item in row['Usage'] if item and item.lower() != 'none']
                st.markdown("##### Excerpt")
                st.write(row[freeform_col])
                if usage_items:
                    st.markdown("##### Usage")
                    pills_html = "".join(
                            f"<span style='display:inline-block;background-color:#E7F4FF;color:#125E9E;border-radius:20px;padding:4px 10px;margin:2px;font-size:0.95rem;'>{item}</span>"
                        for item in usage_items
                    )
                    st.markdown(pills_html, unsafe_allow_html=True)
                else:
                    st.caption("*No usage found*")
                st.write("")

                st.checkbox(
                    "Add to shortlist",
                    key=f"shortlist_{idx}"
                )

        # Shortlist summary and download (manual)
        shortlisted = [
            i for i in filtered_df.index
            if st.session_state.get(f"shortlist_{i}", False)
        ]
        st.sidebar.markdown(f"**Manual Shortlisted:** {len(shortlisted)}")
        if shortlisted:
            csv = df.loc[shortlisted].to_csv(index=False).encode('utf-8')
            st.sidebar.download_button(
                "Download Manual Shortlist", csv, "shortlist.csv", "text/csv"
            )


    ## ------------ INSIGHTS TAB -----------

    with tab2:
        st.write("")

        col1, col2, col3 = st.columns(3)
        col1.metric("Avg. Word Count", f"{df['word_count'].mean().round(1)}")
        col2.metric("Median N.I", df['necessity_index'].median().round(2))
        col3.metric("Total Applications", len(df))
        st.html("<br>")

        st.subheader("Necessity Index (NI) Distribution")
        st.write("")
        st.write("")
        # Histogram of necessity index colored by priority labels
        chart = alt.Chart(df).mark_bar().encode(
            x=alt.X('necessity_index:Q', bin=alt.Bin(maxbins=20), title='Necessity Index'),
            y='count()',
            color=alt.Color(
                'priority:N',
                scale=alt.Scale(
                    domain=['low', 'medium', 'high', 'priority'],
                    range=['#a7d6fd', '#FFA500', '#FF5733', '#FF0000']
                ),
                legend=alt.Legend(title='Priority')
            )
        )
        st.altair_chart(chart, use_container_width=True)
        st.dataframe(df, hide_index=True)
