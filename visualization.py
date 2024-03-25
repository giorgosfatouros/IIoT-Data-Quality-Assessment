import streamlit as st
from utils import visualize_data_quality, visualize_sensor_data

def show():
    st.title('Data Visualization')

    if 'tags' in st.session_state:
        tags = st.session_state['tags']
        with st.expander("Show/Hide Data Description"):
            st.dataframe(tags)

    if 'readings' in st.session_state:
        readings = st.session_state['readings']

        # Perform any additional processing if needed
        # For example, resampling could be dynamic based on user input
        freq = st.selectbox('Select Resampling Frequency:', options=['30min', 'h', 'd', 'W'], index=0)  # Hourly, Daily, Weekly
        resampled_df = readings.resample(freq).mean()

        # Visualization
        visualize_sensor_data(resampled_df)

    else:
        st.write("No sensor data available. Please upload and process data in the Data Loading page.")

