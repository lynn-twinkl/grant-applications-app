import pandas as pd
import plotly.express as px

def plot_hist(df: pd.DataFrame, col_to_plot: str, bins: int, height: int = 500):

    plt = px.histogram(
            df,
            x=col_to_plot,
            nbins=bins,
            color_discrete_sequence=['#646DEF']
            )

    plt.update_layout(
            bargap=0.1,
            height=height
            )

    return plt
