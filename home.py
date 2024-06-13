import streamlit as st


def show():
    st.title('IIoT Data Quality Assessment')

    st.write("""
    This application is designed to help users assess and improve the quality of data collected from Industrial Internet of Things (IIoT) devices. It provides tools for loading data, visualizing it, and analyzing data quality metrics.

    ## How to Use This App
    - Navigate to the **Data Loading** page to upload your equipment descriptions and sensor data. You can visualize the uploaded data and send it to a database for storage.
    - The **Data Quality** page offers visualizations and metrics to assess the quality of your IIoT data, helping you identify areas for improvement.

    ## Features
    - **Data Loading and Visualization**: Upload and visualize your IIoT data.
    - **Data Storage**: Easily store your data in LXS database for overall data analysis.
    - **Quality Metrics Visualization**: Analyze your data with various quality metrics to ensure high data integrity.

    Navigate using the sidebar to start using the app.
    """)
