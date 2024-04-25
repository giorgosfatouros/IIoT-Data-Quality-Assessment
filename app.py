import streamlit as st
from data_loading import show as show_data_loading
from data_quality import show as show_data_quality
from visualization import show as show_data_visuals
from home import show as show_home

logo_url = "./static/logo-azul.svg"
st.set_page_config(page_title="FAME | IIoT Data Quality", page_icon="⚙️",  layout="wide")

st.sidebar.image(logo_url, width=200)
st.sidebar.title('Navigation')

page = st.sidebar.radio('Go to', ['Home', 'Data Loading', 'Data Visualization', 'Data Quality'])


if page == 'Home':
    show_home()
elif page == 'Data Loading':
    show_data_loading()
elif page == 'Data Visualization':
    show_data_visuals()
elif page == 'Data Quality':
    show_data_quality()