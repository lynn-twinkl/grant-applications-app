################################
# CONFIGURATION
################################

import streamlit as st
import pandas as pd
import hashlib
import joblib
from io import BytesIO
from umap import UMAP
from hdbscan import HDBSCAN

import os

from streamlit_extras.metric_cards import style_metric_cards
from streamlit_extras.add_vertical_space import add_vertical_space

# ---- FUNCTIONS ----

from src.extract_usage import extract_usage
from src.necessity_index import compute_necessity, index_scaler, qcut_labels
from src.column_detection import detect_freeform_col, detect_id_col, detect_school_type_col
from src.shortlist import shortlist_applications
from src.twinkl_originals import find_book_candidates
from src.preprocess_text import normalise_text 
import src.models.topic_modeling_pipeline as topic_modeling_pipeline
from src.px_charts import plot_histogram, plot_topic_countplot
from typing import Tuple

style_metric_cards(box_shadow=False, border_left_color='#E7F4FF',background_color='#E7F4FF', border_size_px=0, border_radius_px=6)

##################################
# CACHED PROCESSING FUNCTIONS
##################################

# -----------------------------------------------------------------------------
# Heavy processing (IO + NLP) is cached to avoid re‑executing when the UI state
# changes. The function only re‑runs if the **file contents** change.
# -----------------------------------------------------------------------------

@st.cache_resource
def load_heartfelt_predictor():
    model_path = os.path.join("src", "models", "heartfelt_pipeline.joblib")
    return joblib.load(model_path)

@st.cache_resource
def load_embeddings_model():
    return topic_modeling_pipeline.load_embedding_model('all-MiniLM-L12-v2')

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
    id_col = detect_id_col(df_orig)
    school_type_col = detect_school_type_col(df_orig)
    print(id_col)

    df_orig = df_orig[df_orig[freeform_col].notna()]

    #Word Count
    df_orig['word_count'] = df_orig[freeform_col].fillna('').str.split().str.len()

    # Compute Necessity Scores
    scored = df_orig.join(df_orig[freeform_col].apply(compute_necessity))
    scored['necessity_index'] = index_scaler(scored['necessity_index'].values)
    scored['priority'] = qcut_labels(scored['necessity_index'])

    # Find Twinkl Originals Candidates
    scored['book_candidates'] = find_book_candidates(scored, freeform_col, school_type_col)

    # Label Heartfelt Applications
    scored['clean_text'] = scored[freeform_col].map(normalise_text)
    model = load_heartfelt_predictor()
    scored['is_heartfelt'] = model.predict(scored['clean_text'].astype(str))


    
    # Usage Extraction
    docs = df_orig[freeform_col].to_list()
    scored['usage'] = extract_usage(docs)

    return scored, freeform_col, id_col

# -----------------------------------------------------------------------------
# Derivative computations that rely only on the processed DataFrame are also
# cached. These are lightweight but still benefit from caching because this
# function might be called multiple times during widget interaction.
# -----------------------------------------------------------------------------


@st.cache_data(show_spinner=True)
def compute_shortlist(df: pd.DataFrame) -> pd.DataFrame:
    """Pre‑compute shortlist_score for all rows (used for both modes)."""
    return shortlist_applications(df, k=len(df))

@st.cache_resource(show_spinner=True)
def run_topic_modeling():

    try:

        # ------- 1. Tokenize texts into sentences -------
        nlp = topic_modeling_pipeline.load_spacy_model(model_name='en_core_web_sm')

        sentences = []
        mappings = []

        for idx, application_text in df[freeform_col].dropna().items():
            for sentence in topic_modeling_pipeline.spacy_sent_tokenize(application_text):
                sentences.append(sentence)
                mappings.append(idx)


        # -------- 2. Generate embeddings -------

        embeddings_model = load_embeddings_model()
        embeddings = embeddings_model.encode(sentences, show_progress_bar=True)

        # -------- 3. Topic Modeling --------

        umap_model = UMAP(n_neighbors=7, n_components=5, min_dist=0.0, metric='cosine', random_state=42)
        hdbscan_model = HDBSCAN(min_cluster_size=10, metric='euclidean', cluster_selection_method='eom', prediction_data=True)

        # --------- 4. Perform Topic Modeling ---------

        topic_model, topics, probs = topic_modeling_pipeline.bertopic_model(sentences, embeddings, embeddings_model, umap_model, hdbscan_model)

        topic_modeling_pipeline.ai_labels_to_custom_name(topic_model) 

        return topic_model, topics, probs, mappings

    except Exception as e:
        st.error(f"Topic modeling failed: {e}")
        st.code(traceback.format_exc())  # Shows the full error in a nice code box
        return None, None, None, None



################################
# MAIN APP SCRIPT
################################

st.title("🪷 Community Collections Helper")

uploaded_file = st.file_uploader("Upload grant applications file for analysis", type='csv', label_visibility='collapsed')


# ========== FINGERPRINTING CURRENT FILE ==========
# This helps avoid reruns of certain functions as
# long as the file stays the same

if uploaded_file is not None:
    raw = uploaded_file.read()
    file_hash = hashlib.md5(raw).hexdigest()
    st.session_state["current_file_hash"] = file_hash
else:
    raw = None
    st.session_state.pop("current_file_hash", None)

if raw is None:
    st.stop()

## ====== DATA PROCESSING ======

df, freeform_col, id_col = load_and_process(raw) # from cached function
topic_model, topics, probs, mappings = run_topic_modeling() # from cached function

if topic_model is not None:
    label_map = (topic_model
                 .get_topic_info()
                 .set_index("Topic")["CustomName"]
                 .to_dict())
    df = topic_modeling_pipeline.attach_topics(df, mappings, topics, label_map, col="topics")
else:
    st.warning("Topics could not be generated; continuing without them.")


topics_df = topic_model.get_topic_info()
topics_df = topics_df[topics_df['Topic'] > -1]
topics_df.drop(columns=['Name', 'OpenAI'], inplace=True)
cols_to_move = ['Topic','CustomName']
topics_df = topics_df[cols_to_move + [col for col in topics_df.columns if col not in cols_to_move]]
topics_df.rename(columns={'CustomName':'Topic Name', 'Topic':'Topic Nr.'}, inplace=True)



book_candidates_df = df[df['book_candidates'] == True]

###############################
#         SIDE PANNEL         #
###############################

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

    ## --- Dataframe To Filter ---
    options = ['All applications', 'Not shortlisted'] 
    selected_view = st.pills('Choose data to filter', options, default='Not shortlisted')
    st.write("")

    ## --- Necessity Index Filtering ---
    min_idx = float(df['necessity_index'].min())
    max_idx = float(df['necessity_index'].max())
    filter_range = st.slider(
        "Necessity Index Range", min_value=min_idx, max_value=max_idx, value=(min_idx, max_idx)
    )
    
    def filter_all_applications(df, auto_short_df, filter_range):
        return df[df['necessity_index'].between(filter_range[0], filter_range[1])]

    def filter_not_shortlisted(df, auto_short_df, filter_range):
        return df[
            (~df.index.isin(auto_short_df.index)) &
            (df['necessity_index'].between(filter_range[0], filter_range[1]))
        ]

    filter_map = {
            'All applications': filter_all_applications,
            'Not shortlisted': filter_not_shortlisted,
            }

    filtered_df = filter_map[selected_view](df, auto_short_df, filter_range)

    ## -------- Topic Filtering -------

    topic_options = sorted(topics_df['Topic Name'].unique())
    selected_topics = st.multiselect("Filter by Topic(s)", options=topic_options, default=[])

    if selected_topics:
        selected_set = set(selected_topics)
        filtered_df = filtered_df[
            filtered_df['topics'].apply(lambda topic_list: selected_set.issubset(set(topic_list)))
        ]

    st.markdown(f"**Total Applications:** {len(df)}")
    st.markdown(f"**Filtered Applications:** {len(filtered_df)}")

    manual_keys = [k for k in st.session_state.keys() if k.startswith("shortlist_")]
    manually_shortlisted = [int(k.split("_")[1]) for k in manual_keys if st.session_state[k]]

    st.markdown(f"**Manually Shortlisted:** {len(manually_shortlisted)}")
    if manually_shortlisted:
        csv = df.loc[manually_shortlisted].to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download Manual Shortlist",
            data=csv,
            file_name="manual_shortlist.csv",
            mime="text/csv",
            icon="⬇️",
        )


    add_vertical_space(4)
    st.divider()
    st.badge("Version 1.3.0", icon=':material/category:',color='violet')
    st.markdown("""
    :grey[Made with 🩷  by the AI Innovation Team
    Contact: lynn.perez@twinkl.com]
    """)



## ====== CREATE TAB SECTIONS =======
tab1, tab2 = st.tabs(["Shortlist Manager","Insights"])


##################################################
#              SHORTLIST MANAGER TAB             # 
##################################################

with tab1:
    
    ## =========== AUTOMATIC SHORTLIST =========

    st.header("Automatic Shortlist")

    csv_auto = auto_short_df.to_csv(index=False).encode("utf-8")
    all_processed_data = df.to_csv(index=False).encode("utf-8")
    book_candidates = book_candidates_df.to_csv(index=False).encode("utf-8")
    topic_descriptions_csv = topics_df.to_csv(index=False).encode("utf-8")


    csv_options = {
        "Shortlist": (csv_auto, "shortlist.csv"),
        "All Processed Data": (all_processed_data, "all_processed.csv"),
        "Book Candidates": (book_candidates, "book_candidates.csv"),
        "Topic Descriptions": (topic_descriptions_csv, "topic_descriptions.csv"),
    }

    choice = st.selectbox("Select a file for download", list(csv_options.keys()))

    csv_data, file_name = csv_options[choice]


    st.download_button(
        label=f"Download {choice}",
        data=csv_data,
        file_name=file_name,
        mime="text/csv",
        help="This button will download the selected file from above",
        icon="⬇️"

    )

    
    st.write("")
    total_col, shortlistCounter_col, mode_col = st.columns(3)

    total_col.metric("Applications Submitted", len(df))
    shortlistCounter_col.metric("Shorlist Length",  len(auto_short_df))
    mode_col.metric("Mode", mode)

    shorltist_cols_to_show = [
            id_col,
            freeform_col,
            'book_candidates',
            'usage',
            'necessity_index',
            'urgency_score',
            'severity_score',
            'vulnerability_score',
            'shortlist_score',
            'is_heartfelt',
            'topics',
            ]

    st.dataframe(auto_short_df.loc[:, shorltist_cols_to_show], hide_index=True)

    ## ====== APPLICATIONS REVIEW =======

    add_vertical_space(2)
    st.header("Manual Filtering")
    st.info("Use the **side panel** filters to more easily sort through applications that you'd like to review.", icon=':material/info:')

    st.write("")
    if len(filtered_df) > 0:
        st.markdown("#### Filtered Applications")
        for idx, row in filtered_df.iterrows():
            with st.expander(f"Application {int(row[id_col])}"):
                st.write("")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Necessity", f"{row['necessity_index']:.1f}")
                col2.metric("Urgency", f"{int(row['urgency_score'])}")
                col3.metric("Severity", f"{int(row['severity_score'])}")
                col4.metric("Vulnerability", f"{int(row['vulnerability_score'])}")

                st.markdown("##### Excerpt")
                st.write(row[freeform_col])

                # HTML for clean usage items 
                usage_items = [item for item in row['usage'] if item and item.lower() != 'none']
                if usage_items:
                    st.markdown("##### Usage")
                    pills_html = "".join(
                            f"<span style='display:inline-block;background-color:#E7F4FF;color:#125E9E;border-radius:20px;padding:4px 10px;margin:2px;font-size:0.95rem;'>{item}</span>"
                        for item in usage_items
                    )
                    st.html(pills_html)
                else:
                    st.caption("*No usage found*")

                topic_items = [item for item in row['topics'] if item and item.lower() != 'none']
                if topic_items:
                    st.markdown("##### Topics")
                    topic_boxes_html= "".join(
                    f"<span style='display:inline-block;background-color:#ECE0FC;color:#6741B9;border-radius:5px;padding:4px 10px;margin:2px;font-size:0.95rem;'>{item}</span>"
                    for item in topic_items
                    )
                    st.html(topic_boxes_html)
                else:
                    st.caption("_No topics assigned for this application_")


                st.checkbox(
                    "Add to shortlist",
                    key=f"shortlist_{idx}"
                )

    else:
        st.markdown(
            """
            <br>
            <div style="text-align: center; font-size: 1.2em">
                🍂 <span style="color: grey;">No applications matched these filters...</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


#########################################
#              INSIGHTS TAB             #
#########################################

with tab2:


    ## =========== DATA OVERVIEW ==========

    st.header("General Insights")
    add_vertical_space(1)

    col1, col2, col3 = st.columns(3)
    col1.metric("Applications Submitted", len(df))
    col2.metric("Median N.I", df['necessity_index'].median().round(2))
    col3.metric("Avg. Word Count", f"{df['word_count'].mean().round(1)}")

    ## --- NI Distribution Plot ---
    ni_distribution_plt = plot_histogram(df, col_to_plot='necessity_index', bins=50, title='Necessity Index Histogram')
    st.plotly_chart(ni_distribution_plt)


    ## ============= TOPIC MODELING ============

    st.header("Topic Modeling")
    add_vertical_space(1)
    
    ## ------- Display Topics Dataframe ------

    with st.popover("How are topic extracted?", icon="🌱"):

        st.write("""
        **About Topic Modeling**

        We use BERTopic to :primary[**dynamically**] extract the most common topics from the natural language data.

        BERTopic is a machine learning technique that allows us to group documents (in this case, sentences within application letters) based on their semantic similarity and other patterns such as word frequency and placement.

        The table you see below shows you the extracted topics, alongside their top 10 extracted keywords and a small sample of real texts from the applications that demonstrate where the topics came from.

        **Table Info**
        - **Topic Nr.:** The 'id' of the topic.
        - **Topic Name:** This is an AI-generated label based on a few samples of application responses alongside their corresponding keywords.
        - **Representation:** Top 10 keywords that best represent a topic
        - **Representative Docs**: Sample sentences contributing to the topic
        """)
    st.dataframe(topics_df, hide_index=True)

    ## -------- Plot Topics Chart ----------

    topic_count_plot = plot_topic_countplot(topics_df, topic_id_col='Topic Nr.', topic_name_col='Topic Name', representation_col='Representation', height=500, title='Topic Frequency Chart')
    st.plotly_chart(topic_count_plot, use_container_width=True)

    ## --------- User Updates -----------

    if st.session_state.get("topic_toast_shown_for") != st.session_state["current_file_hash"]:
        st.toast(
        """
        **Topic modeling is ready!** View the results on the _Insights_ tab
        """,
        icon='🎉'
        )

        st.session_state["topic_toast_shown_for"] = st.session_state["current_file_hash"]
