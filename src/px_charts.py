import pandas as pd
import plotly.express as px


title_font_size=18
title_font_color='#808393'

def plot_histogram(df: pd.DataFrame, col_to_plot: str, bins: int, height: int = 500, title:str = None):

    plt = px.histogram(
            df,
            x=col_to_plot,
            nbins=bins,
            title=title,
            color_discrete_sequence=['#646DEF']
            )

    plt.update_layout(
            bargap=0.1,
            height=height,
            title_font_size=title_font_size,
            title_font_color=title_font_color
            )

    return plt


# =========== TOPIC DISTRIBUTION CHART  ===========


def plot_topic_countplot(topics_df: pd.DataFrame, topic_id_col: str, topic_name_col: str, representation_col: str, height: int = 500, title:str = None):
    """
    This functions plots a count chart for Bertopic topics,
    extracting the 5 words of each topic's representation
    in order to provide more context
    """

    ## ----- Extract top 5 words ----
    topics_df['top_5_words'] = topics_df[representation_col].apply(lambda x: ", ".join(x[:5]) if isinstance(x, list) else x)

    plt = px.bar(
            topics_df,
            x=topic_id_col,
            y='Count',
            custom_data=["top_5_words", topic_name_col],
            title=title,
            )

    plt.update_xaxes(type='category')

    plt.update_traces(
        marker_color='#EF64B3',
        textposition='outside',
        hovertemplate=(
            'Topic Name: %{customdata[1]}<br>'
            'Frequency: %{y}<br>'
            'Top 5 words: %{customdata[0]}<extra></extra>'
            )
        )

    plt.update_layout(
        uniformtext_minsize=10,
        height=height,
        hoverlabel=dict(
            font_size=13,
            align="left"
        ),
        title_font_size=title_font_size,
        title_font_color=title_font_color

    )



    return plt
