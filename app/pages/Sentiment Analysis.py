import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# from nltk.tokenize import RegexpTokenizer
import streamlit.components.v1 as components
from PIL import Image

# from annotated_text import annotated_text
from bertopic import BERTopic
from streamlit_folium import st_folium, folium_static
import datetime
import folium
from folium.plugins import HeatMap

# st.title('A title with _italics_ :blue[colors] and emojis :sunglasses:')
st.set_page_config(page_icon="chart_with_upwards_trend")


st.title("Sentiment analysis distribution in Paris")


df = pd.read_parquet("artifacts/sentiment_analysis.parquet")
df_plot = df.copy()
df["label"] = df["label"].apply(lambda x: 1 if x == "NEGATIVE" else 0)

with st.sidebar:
    year = st.selectbox("Year", range(2009, 2024), 14)
    month = st.selectbox("Month", range(1, 13), 2)

# st.write(month, year)


df_date = df[(df["month"] == month) & (df["year"] == year)]


if len(df_date) == 0:
    st.warning("Data not available for this time period", icon="⚠️")
else:
    map_obj = folium.Map(
        location=[48.85, 2.34], zoom_start=12.8, tiles="CartoDB positron"
    )
    HeatMap(df_date[["latitude", "longitude", "label"]].values, min_opacity=0.5).add_to(
        map_obj
    )

    folium_static(map_obj)

    with st.sidebar:
        st.warning('Here, the intensity represents negative sentiment')


st.markdown("## Statistics by day of the week")

fig = px.histogram(
    df_plot,
    x="dayname",
    y="listing_id",
    color="label",
    barmode="group",
    histfunc="count",
)

st.plotly_chart(fig, theme=None, use_container_width=True)


st.markdown("## Statistics by day of month")

fig = px.histogram(
    df,
    x="month",
    y="listing_id",
    color="label",
    barmode="group",
    histfunc="count",
)

st.plotly_chart(fig, theme=None, use_container_width=True)
