import streamlit as st
from utils import visualize_overall_data_quality


def calculate_missing_readings(data):
    """
    Calculate the percentage of missing readings (NaNs) and periods of missing readings for each sensor.

    Parameters:
    data (pd.DataFrame): The dataframe containing sensor readings.
    original_freq_sec (int): The frequency of the original data in seconds (default is 10 seconds).

    Returns:
    pd.DataFrame: A dataframe with the percentage of missing readings for each sensor.
    dict: A dictionary with the periods of missing readings for each sensor.
    """
    aggr = st.session_state['AGG_FREQ']
    original_freq_sec = st.session_state['ORIG_FREQ']
    # Calculate the total number of expected readings
    total_duration_sec = (data.index[-1] - data.index[0]).total_seconds()
    expected_readings = total_duration_sec // original_freq_sec

    sensor_columns = [col for col in data.columns if 'count_' in col and '_isvalid' not in col]

    # Rename sensor columns to remove 'count_'
    renamed_columns = {col: col.replace('count_col', '') for col in sensor_columns}
    sensor_columns = renamed_columns.values()
    data = data.rename(columns=renamed_columns)

    missing_percentages = (expected_readings - data[sensor_columns].sum()) / expected_readings * 100
    missing_percentages = missing_percentages.reset_index()
    missing_percentages.columns = ['sensor', 'missing_percentage']

    missing = (expected_readings - data[sensor_columns].sum())
    missing = missing.reset_index()
    missing.columns = ['sensor', 'missing']

    missing_intervals = {}
    for sensor in sensor_columns:
        missing_timestamps = data[data[sensor] < aggr].index
        missing_intervals[sensor] = identify_intervals(missing_timestamps, aggr * original_freq_sec)

    return missing, missing_percentages, missing_intervals


def identify_intervals(timestamps, freq_sec):
    """
    Identify contiguous intervals from a list of timestamps.

    Parameters:
    timestamps (list): List of timestamps where readings are missing.
    freq_sec (int): The frequency of the original data in seconds.

    Returns:
    list: A list of tuples, each representing a start and end of a missing interval.
    """
    if timestamps.empty:
        return []

    intervals = []
    start = timestamps[0]
    end = timestamps[0]

    for i in range(1, len(timestamps)):
        if (timestamps[i] - end).total_seconds() <= freq_sec:
            end = timestamps[i]
        else:
            intervals.append((start, end))
            start = timestamps[i]
            end = timestamps[i]

    intervals.append((start, end))  # Add the last interval
    return intervals


def show():
    st.title('Missing Values Analysis')
    st.write("This page analyzes the missing readings from the raw sensor data, which are aggregated "
             "from the original frequency for fast and efficient analysis. The focuses on detecting and understanding "
             "the data completeness of the original dataset.")

    if 'tags' in st.session_state:
        tags = st.session_state['tags']
        with st.expander("Show/Hide Data Description"):
            st.dataframe(tags)

    if 'readings' in st.session_state:
        readings = st.session_state['raw_readings']

        # Calculate missing readings
        missing_readings_df, missing_percentages_df, missing_intervals = calculate_missing_readings(readings)

        # Visualization
        visualize_overall_data_quality(missing_readings_df, missing_percentages_df, missing_intervals)

    else:
        st.warning("No sensor data available. Please upload and process data in the Data Loading page.")


if __name__ == "__main__":
    show()
