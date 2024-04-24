from datetime import datetime
import streamlit as st
import pandas as pd
from utils import preprocess_sensor_data
from model import load_config, construct_sensor_data_class, get_session

config = load_config()
if 'SensorData' not in globals():
    SensorData = construct_sensor_data_class(config)
engine = setup_database(config)




def show():
    st.title('Data Loading and Storage')

    if 'tags' not in st.session_state:
        uploaded_file2 = st.file_uploader("**Upload Sensor and Equipment/Machine descriptions**", type=['csv', 'xlsx'])
        if uploaded_file2 is not None:
            tags = pd.read_excel(uploaded_file2, header=0)
            # tags = tags.iloc[:, 1:]
            tags.columns = [col.lower().replace(' ', '_') for col in tags.columns]
            with st.expander("Show/Hide Data Description"):
                st.dataframe(tags)
            selected_machine = st.selectbox('Select a Machine:', tags['machine_group'].unique())
            st.write(f'You have selected Equipment: **{selected_machine}**')
            machine_tags = tags[tags['machine_group'].str.contains(selected_machine[1:], case=False, na=False)].reset_index(
                drop=True)
            st.session_state['tags'] = machine_tags

            st.write(f'**{machine_tags.shape[0]}** Sensors are monitoring the **{selected_machine}** equipment')

    if 'readings' not in st.session_state:
        uploaded_file = st.file_uploader("**Upload Sensor Data**", type=['csv', 'xlsx'])
        if uploaded_file is not None:
            with st.spinner('Processing...'):
                readings = pd.read_excel(uploaded_file, sheet_name='Parameters')
                readings = preprocess_sensor_data(readings)
                st.session_state['readings'] = readings
            with st.expander("Show/Hide Data"):
                st.dataframe(readings)

            persist_data = st.checkbox("Persist data in the database for overall analytics?")
            if persist_data:
                session = get_session(engine)
                load_data_to_db(readings, session, SensorData)

                st.success("Data successfully saved to the database!")
                st.success("Data successfully prepared for visualization. Please proceed to the **Visualization** "
                           "page to explore your data.")
            else:
                st.success("Data successfully prepared for visualization. Please proceed to the **Visualization** "
                           "page to explore your data.")

            # Resampling frequency
            # freq = '60T'
            # # Resample non-boolean columns (assuming mean as aggregation function)
            # resampled_non_bool = readings.select_dtypes(exclude='bool').resample(freq).mean()
            #
            # # Resample boolean columns using apply method
            # resampled_bool = readings.select_dtypes(include='bool').resample(freq).apply(
            #     lambda x: x.any())  # or x.all()

            # Combine the resampled data
            # readings = pd.concat([resampled_non_bool, resampled_bool], axis=1)
