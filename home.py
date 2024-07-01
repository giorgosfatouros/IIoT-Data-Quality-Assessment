import streamlit as st


def show():
    st.title('IIoT Data Quality Assessment')

    st.write("This application is designed to help you analyze and assess the quality of data collected from Industrial  Internet of Things (IIoT) devices. Below are the main features and instructions on how to use the app.")
    
    st.markdown("""
    ## Features 
    - **Data Loading**: Import raw data or connect to the LeanXscale (LXS) database via Kafka for real-time data streaming. 
    - **LeanXscale Integration**: Leverage online aggregates and incremental analytics for fast and efficient data processing. 
    - **Data Annotation**: The aggregated data are automatically annotated based on the nominal sensor values and can be exported to for further exploitation. 
    - **Data Visualization**: Visualize the loaded and aggregated data to understand its structure and quality. 
    - **Missing Values Analysis**: Detect and handle missing values in the raw sensor dataset, utilizing aggregated data. 
    - **Invalid Values Analysis**: Identify and analyze invalid readings or alarms from your sensors. 
    - **Data Quality**: Perform comprehensive data quality assessments, including metrics and visualizations.

    ## How to Use This App
    1. **Data Loading**: Navigate to the **Data Loading** page to select the data table (machine) for analysis.
    2. **Data Visualization**: Use the **Data Visualization** page to explore the data through various visualizations.
    3. **Missing Values Analysis**: Go to the **Missing Values Analysis** to get insights for missing values into the original/raw data.
    4. **Invalid Values Analysis**: The **Invalid Values Analysis** page helps you identify and understand invalid readings from your sensor data.
    5. **Data Quality**: Access the **Data Quality** page for a detailed assessment of your data's quality, including completeness, accuracy, and consistency.

    Navigate using the sidebar to start using the app. Each page provides specific tools and visualizations for data quality assessment of IoT sensor data.
    """)
