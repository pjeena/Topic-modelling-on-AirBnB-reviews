import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
#from nltk.tokenize import RegexpTokenizer
import streamlit.components.v1 as components
from PIL import Image
#from annotated_text import annotated_text
from bertopic import BERTopic



st.set_page_config(
    page_title="Streamlit Dashboard",
    layout="wide",
    page_icon="💹",
    initial_sidebar_state="expanded",
)

st.title("Topic Modeling and Sentiment analysis on AirBnB reviews")
st.write('*This is a dashboard based on the [AirBnB reviews](http://insideairbnb.com/get-the-data) \
         in the city of Paris. 50000 reviews regarding hotel stays were analysed using BertTopic*')

st.markdown("""
        <style>
               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 0rem;
                    padding-left: 5rem;
                    padding-right: 5rem;
                }
        </style>
        """, unsafe_allow_html=True)

#################################################### Calculations #############################################################################
@st.cache_data(experimental_allow_widgets=True)
def load_model():

    topic_model = BERTopic.load("model/model_dir")
    return topic_model

topic_model = load_model()

#df = pd.read_parquet('data/paris_reviews_preprocessed.parquet')
#################################################### Calculations #############################################################################


### top row

#st.markdown("## ")

# st.markdown("<hr/>", unsafe_allow_html=True)

st.markdown(
    """<hr style="height:5px;border:none;color:#333;background-color:#333;" /> """,
    unsafe_allow_html=True,
)


#st.dataframe(df.head(), hide_index=True)

st.markdown(
    """<hr style="height:5px;border:none;color:#333;background-color:#333;" /> """,
    unsafe_allow_html=True,
)


st.markdown(
    "## Top 30 frequent topics"
)

st.dataframe(topic_model.get_topic_info()[1:30][['Topic', 'Count', 'Name', 'Representation']],use_container_width=True)


st.markdown(
    "## 2D representation of the topics"
)
st.plotly_chart( topic_model.visualize_topics(top_n_topics=30),theme='streamlit', use_container_width=True )



st.markdown(
    "## Topic similarities"
)
st.plotly_chart( topic_model.visualize_heatmap(top_n_topics=30) ,theme='streamlit', use_container_width=True )



st.markdown(
    "## Visualizing individual terms of the top 8 topics"
)
st.plotly_chart( topic_model.visualize_barchart(),theme='streamlit', use_container_width=True )





