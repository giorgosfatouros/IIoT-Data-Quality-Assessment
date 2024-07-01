import streamlit as st
from utils import display_general_info, display_missing_values, display_outliers, check_for_duplicates, \
    check_timestamp_consistency, check_data_accuracy, check_data_completeness, calculate_correlation_matrix, \
    display_correlation_report, analyze_strong_correlations, display_strong_correlations


def show():
    st.title('Data Quality Assessment Report')

    if 'readings' in st.session_state:
        readings = st.session_state['readings']
        readings = readings[[col for col in readings.columns if 'alarm' not in col]]

        st.header("General Information")
        if 'tags' in st.session_state:
            tags = st.session_state['tags']
            with st.expander("**Show/Hide Data Description**"):
                st.dataframe(tags)
        display_general_info(readings)

        st.header("Data Consistency Checks")
        check_for_duplicates(readings)
        check_timestamp_consistency(readings)

        st.header("Data Completeness")
        check_data_completeness(readings)
        display_missing_values(readings)

        st.header("Outliers")
        display_outliers(readings)

        st.header("Data Accuracy")
        check_data_accuracy(readings, tags)

        st.header("Correlation Report")
        corr_matrix = calculate_correlation_matrix(readings)
        display_correlation_report(corr_matrix)
        strong_correlations = analyze_strong_correlations(corr_matrix, threshold=0.7)
        display_strong_correlations(strong_correlations)

    else:
        st.warning("No data available for analysis. Please upload data in the Data Loading page.")


