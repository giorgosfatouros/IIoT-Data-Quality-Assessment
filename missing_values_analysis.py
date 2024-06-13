import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from utils import visualize_overall_data_quality


def calculate_missing_readings(data, original_freq_sec=10, agg_interval_sec=3600):
    """
    Calculate the number of missing readings (NaNs) and their percentage for each sensor per hour.

    Parameters:
    data (pd.DataFrame): The dataframe containing sensor readings.
    original_freq_sec (int): The frequency of the original data in seconds (default is 10 seconds).
    agg_interval_sec (int): The aggregation interval in seconds (default is 3600 seconds for 1 hour).

    Returns:
    pd.DataFrame: A dataframe with the number and percentage of missing readings for each sensor per hour.
    """
    expected_readings_per_hour = agg_interval_sec // original_freq_sec
    sensor_columns = [col for col in data.columns if 'count_' in col and '_isvalid' not in col]
    missing_readings = {}
    missing_percentages = {}

    for sensor in sensor_columns:
        missing_col = f"{sensor}_missing"
        percentage_col = f"{sensor}_missing_percentage"
        data[missing_col] = expected_readings_per_hour - data[sensor]
        data[percentage_col] = (data[missing_col] / expected_readings_per_hour) * 100
        missing_readings[missing_col] = data[missing_col]
        missing_percentages[percentage_col] = data[percentage_col]

    missing_readings_df = pd.DataFrame(missing_readings)
    missing_percentages_df = pd.DataFrame(missing_percentages)
    return missing_readings_df, missing_percentages_df


def visualize_data_quality(missing_readings_df, missing_percentages_df):
    # Check if there is data to plot
    if not missing_readings_df.empty:
        # Plotting the total missing readings per sensor
        plt.figure(figsize=(14, 7))
        missing_totals = missing_readings_df.sum()
        sns.barplot(x=missing_totals.index, y=missing_totals.values)
        plt.title("Total Missing Readings per Sensor")
        plt.xlabel("Sensor")
        plt.ylabel("Total Missing Readings")
        plt.xticks(rotation=90)
        st.pyplot(plt)

        # Plotting the heatmap of missing readings
        plt.figure(figsize=(14, 7))
        sns.heatmap(missing_readings_df.isnull(), cbar=False, cmap='viridis')
        plt.title("Heatmap of Missing Readings")
        plt.xlabel("Sensors")
        plt.ylabel("Time")
        st.pyplot(plt)

    if not missing_percentages_df.empty:
        # Plotting the total missing readings percentages per sensor
        plt.figure(figsize=(14, 7))
        missing_percentage_totals = missing_percentages_df.mean()
        sns.barplot(x=missing_percentage_totals.index, y=missing_percentage_totals.values)
        plt.title("Average Missing Readings Percentage per Sensor")
        plt.xlabel("Sensor")
        plt.ylabel("Average Missing Percentage (%)")
        plt.xticks(rotation=90)
        st.pyplot(plt)


def show():
    st.title('Missing Values Analysis')

    if 'tags' in st.session_state:
        tags = st.session_state['tags']
        with st.expander("Show/Hide Data Description"):
            st.dataframe(tags)

    if 'readings' in st.session_state:
        readings = st.session_state['raw_readings']

        # Calculate missing readings
        missing_readings_df, missing_percentages_df = calculate_missing_readings(readings)

        # Visualization
        visualize_overall_data_quality(missing_readings_df, missing_percentages_df)

        # Show overall statistics
        st.subheader("Overall Statistics")
        st.write("### Missing Readings Summary")
        st.dataframe(missing_readings_df.describe())
    else:
        st.warning("No sensor data available. Please upload and process data in the Data Loading page.")


if __name__ == "__main__":
    show()
