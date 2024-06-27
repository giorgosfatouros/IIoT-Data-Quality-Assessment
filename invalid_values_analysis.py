import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


def calculate_invalid_readings(data, original_freq_sec=10, agg_interval_sec=3600):
    """
    Calculate the number and percentage of invalid readings for each sensor per hour.

    Parameters:
    data (pd.DataFrame): The dataframe containing sensor readings.
    original_freq_sec (int): The frequency of the original data in seconds (default is 10 seconds).
    agg_interval_sec (int): The aggregation interval in seconds (default is 3600 seconds for 1 hour).

    Returns:
    pd.DataFrame: A dataframe with the number and percentage of invalid readings for each sensor per hour.
    """
    if 'timestamp' in data.columns:
        data.set_index('timestamp', inplace=True)
    elif not isinstance(data.index, pd.DatetimeIndex):
        raise KeyError("The dataframe must have a datetime index or a 'timestamp' column.")

    expected_readings_per_hour = agg_interval_sec // original_freq_sec
    sensor_columns = [col for col in data.columns if col.startswith('count_') and not col.endswith('_isvalid')]

    invalid_readings = {}
    invalid_percentages = {}

    for sensor in sensor_columns:
        base_sensor = sensor.replace('count_', '')
        valid_col = f"count_{base_sensor}_isvalid"
        invalid_col = f"count_{base_sensor}_invalid"
        percentage_col = f"count_{base_sensor}_invalid_percentage"

        data[invalid_col] = data[sensor] - data[valid_col]
        data[percentage_col] = (data[invalid_col] / expected_readings_per_hour) * 100

        invalid_readings[invalid_col] = data[invalid_col]
        invalid_percentages[percentage_col] = data[percentage_col]

    invalid_readings_df = data[list(invalid_readings.keys())].reset_index()
    invalid_percentages_df = data[list(invalid_percentages.keys())].reset_index()
    return invalid_readings_df, invalid_percentages_df


def isvalid_df(data, original_freq_sec=10, agg_interval_sec=3600):
    """
    Calculate the number and percentage of invalid readings for each sensor per hour.

    Parameters:
    data (pd.DataFrame): The dataframe containing sensor readings.
    original_freq_sec (int): The frequency of the original data in seconds (default is 10 seconds).
    agg_interval_sec (int): The aggregation interval in seconds (default is 3600 seconds for 1 hour).

    Returns:
    pd.DataFrame: A dataframe with the percentage of invalid readings for each sensor per hour.
    """
    expected_readings_per_hour = agg_interval_sec // original_freq_sec
    sensor_columns = [col for col in data.columns if '_isvalid' in col]
    print(sensor_columns)
    print(expected_readings_per_hour)

    return data[sensor_columns] / expected_readings_per_hour


def plot_invalid_over_mean(data, invalid_data, sensor):

    # # Ensure timestamp is datetime
    # if not pd.api.types.is_datetime64_any_dtype(data.index):
    #     data.index = pd.to_datetime(data.index)
    # if not pd.api.types.is_datetime64_any_dtype(invalid_data['timestamp']):
    #     invalid_data['timestamp'] = pd.to_datetime(invalid_data['timestamp'])
    #
    # # Set timestamp as index for invalid_data
    # invalid_data.set_index('timestamp', inplace=True)

    # Plot the mean readings time series
    mean_series = data[f'sum_{sensor}'] / data[f'count_{sensor}']
    with st.expander(f'Mean Readings and Invalid Readings for {sensor[3:]}', expanded=False):
        plt.figure(figsize=(10, 3))
        plt.plot(mean_series, label='Mean Readings', color='blue')

        # Plot the invalid data points
        invalid_points = invalid_data[invalid_data[f'count_{sensor}_isvalid'] > 2]
        plt.scatter(invalid_points.index, mean_series.loc[invalid_points.index], color='red', label='Invalid Readings')

        plt.xlabel('Time')
        plt.ylabel('Mean Reading')
        plt.legend()
        st.pyplot(plt)


def visualize_invalid_data(invalid_readings_df, data):
    sensors = [col.replace('count_', '').replace('_isvalid', '') for col in invalid_readings_df.columns if
               '_isvalid' in col]

    for sensor in sensors:
        plot_invalid_over_mean(data, invalid_readings_df, sensor)


def show():
    st.title('Invalid Values Analysis')

    if 'tags' in st.session_state:
        tags = st.session_state['tags']
        with st.expander("Show/Hide Data Description"):
            st.dataframe(tags)

    if 'readings' in st.session_state:
        readings = st.session_state['raw_readings']

        # Calculate invalid readings
        valid_readings_df = isvalid_df(readings)

        # Visualization
        visualize_invalid_data(valid_readings_df, readings)

        # Show overall statistics
        st.subheader("Overall Statistics")
        st.write("### Invalid Readings Summary")
        valid_readings_df.rename(columns=lambda x: x.replace('count_', '').replace('_isvalid', ''), inplace=True)
        st.dataframe((valid_readings_df*360).describe())
    else:
        st.warning("No sensor data available. Please upload and process data in the Data Loading page.")


if __name__ == "__main__":
    show()
