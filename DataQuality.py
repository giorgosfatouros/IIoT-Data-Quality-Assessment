import streamlit as st
import pandas as pd
from io import StringIO
import numpy as np

from utils import preprocess_sensor_data

st.title('IIoT Data Quality Assessment')

DATE_COLUMN = 'date/time'
DATA_URL = ('https://s3-us-west-2.amazonaws.com/'
         'streamlit-demo-data/uber-raw-data-sep14.csv.gz')


def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data


uploaded_file = st.file_uploader("Upload Equipment descriptions")
if uploaded_file is not None:
    # Can be used wherever a "file-like" object is accepted:
    tags_desc = pd.read_excel(uploaded_file)
    tags_desc.columns = [col.lower().replace(' ', '_') for col in tags_desc.columns]
    tags_desc.rename(columns={'tag_id': 'tag'}, inplace=True)
    tags_desc['tag'] = tags_desc['tag'].str.replace(".pv", '')
    # tags_desc = tags_desc[['tag', 'tag_description']]
    # st.write(tags_desc)

    # Using an expander to show/hide the selected data
    with st.expander("Show/Hide Data"):
        # Optionally, you can show the entire DataFrame or any other information here
        st.dataframe(tags_desc)  # This line displays the DataFrame inside the expander

    # Dropdown menu
    selected_machine = st.selectbox('Select a Machine:', tags_desc['equipment'].unique())
    st.write(f'You selected Equipment: {selected_machine}')


uploaded_file = st.file_uploader("Upload Sensor Data")
if uploaded_file is not None:
    # Can be used wherever a "file-like" object is accepted:
    readings = pd.read_excel(uploaded_file, sheet_name='Parameters')
    readings = preprocess_sensor_data(readings)
    # tags_desc = tags_desc[['tag', 'tag_description']]
    # st.write(tags_desc)

    # Using an expander to show/hide the selected data
    with st.expander("Show/Hide Data"):
        # Optionally, you can show the entire DataFrame or any other information here
        st.dataframe(readings)  # This line displays the DataFrame inside the expander

    ##### send to LXS ####


