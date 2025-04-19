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
from functions.column_detection import detect_freeform_answer_col
from functions.shortlist import shortlist_applications
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
    #Word count
    df_orig['word_count'] = df_orig[freeform_col].fillna('').str.split().str.len()
    # Compute necessity scores
    scored = df_orig.join(df_orig[freeform_col].apply(compute_necessity))
    scored['necessity_index'] = index_scaler(scored['necessity_index'].values)
    scored['priority'] = qcut_labels(scored['necessity_index'])
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

    ## ---- PROCESSED DATA (CACHED) ----

    df, freeform_col = load_and_process(raw)

    ## ---- INTERACTIVE FILTERING & REVIEW INTERFACE ----

    st.sidebar.title("Shortlist Mode")
    with st.sidebar:
        mode = st.segmented_control(
                "Select one option",
                options=["strict", "generous"],
                default="strict",
                )

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

    tab1, tab2 = st.tabs(["Shortlist Manager","Insights"])

    with tab1:
        # Automatic Shortlisting Controls
        st.header("✨ Automatic Shortlist")
        

        st.markdown("Here's your **automatically genereated shortlist!** If you'd like to manually add additional applications, you may do so on the section below!")
        # Full scores for threshold calculation
        scored_full = shortlist_applications(filtered_df, k=len(filtered_df))
        quantile_map = {"strict": 0.75, "generous": 0.5}
        threshold_score = scored_full["auto_shortlist_score"].quantile(quantile_map[mode])
        auto_short = shortlist_applications(filtered_df, threshold=threshold_score)
        csv_auto = auto_short.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Shortlist",
            csv_auto,
            "auto_shortlist.csv",
            "text/csv"
        )
        st.markdown("#### Shortlist Preview")
        freeform_col_index = auto_short.columns.get_loc(freeform_col)
        st.dataframe(auto_short.iloc[:, freeform_col_index:], hide_index=True)
        # Review applications
        st.header("🌸 Manual Filtering")
        st.markdown(
                """
                Use the side panel filters to manually review applications and add them to your shortlist.
                All **only filtered results** will show up here.
                """
                )
        for idx, row in filtered_df.iterrows():
            with st.expander(f"Application \#{idx}"):
                st.write("")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Necessity", f"{row['necessity_index']:.1f}")
                col2.metric("Urgency", f"{int(row['urgency_score'])}")
                col3.metric("Severity", f"{int(row['severity_score'])}")
                col4.metric("Vulnerability", f"{int(row['vulnerability_score'])}")
                style_metric_cards(box_shadow=False, border_left_color='#E7F4FF',background_color='#E7F4FF', border_size_px=0, border_radius_px=6)
                # Clean usage items
                usage_items = [item for item in row['Usage'] if item and item.lower() != 'none']
                st.markdown("##### Excerpt")
                st.write(row[freeform_col])
                if usage_items:
                    st.markdown("##### Usage")
                    # Display usage items as colored pills
                    pills_html = "".join(
                            f"<span style='display:inline-block;background-color:#E7F4FF;color:#125E9E;border-radius:20px;padding:4px 10px;margin:2px;font-size:0.95rem;'>{item}</span>"
                        for item in usage_items
                    )
                    st.markdown(pills_html, unsafe_allow_html=True)
                else:
                    st.caption("*No usage found*")
                st.write("")
                # Shortlist checkbox
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


    with tab2:
        st.write("")

        col1, col2, col3 = st.columns(3)
        col1.metric("Avg. Word Count", f"{df['word_count'].mean().round(1)}")
        col2.metric("Median N.I", df['necessity_index'].median())
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
