import streamlit as st
from utils import visualize_data_quality, visualize_sensor_data


def show():
    st.title('Data Visualization')

    if 'tags' in st.session_state:
        tags = st.session_state['tags']
        with st.expander("Show/Hide Data Description"):
            st.dataframe(tags)

    if 'sensors' in st.session_state:
        sensors = st.session_state['sensors']
        # Allow user to select columns
        selected_sensors = st.sidebar.multiselect(
            "Select Columns for Analysis",
            options=sensors,
            default=[]
        )
        selected_columns = [s for s in selected_sensors]

        if 'readings' in st.session_state:
            readings = st.session_state['readings']

            # Check if any columns are selected
            if selected_columns:
                # Visualization
                visualize_sensor_data(readings, selected_columns, tags)
            else:
                st.warning("Please select at least one column for analysis.")
        else:
            st.warning("No sensor data available. Please upload and process data in the Data Loading page.")
