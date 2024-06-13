import pandas as pd
import plotly.express as px
import streamlit as st
# from pygwalker.api.streamlit import StreamlitRenderer, init_streamlit_comm
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.fft import fft
import numpy as np
import re


def extract_numbers(s):
    return ''.join(re.findall(r'\d+', s))


def preprocess_sensor_data(df):
    sensors = list(set(col.split('_')[1] for col in df.columns if col.startswith('sum')))

    # Calculate mean value per sensor
    mean_values_per_sensor = {}
    for sensor in sensors:
        sum_col = f'sum_{sensor}'
        count_col = f'count_{sensor}'
        mean_col = f'mean_{sensor}'

        if sum_col in df.columns and count_col in df.columns:
            mean_values_per_sensor[mean_col] = df[sum_col] / df[count_col]

    mean_df = pd.DataFrame(mean_values_per_sensor)
    # df = pd.concat([mean_df, df], axis=1)
    sensors = [s.replace('col', '') for s in sensors]
    mean_df.columns = sensors
    return mean_df, sensors


def display_general_info(df):
    """Display general information about the dataset including date range and data points per sensor."""
    st.subheader('General Dataset Information')

    # Display date range
    if not df.index.empty and px.pd.api.types.is_datetime64_any_dtype(df.index):
        date_range_start = df.index.min().strftime('%Y-%m-%d %H:%M:%S')
        date_range_end = df.index.max().strftime('%Y-%m-%d %H:%M:%S')
        st.markdown(f"📅 **Date Range:** {date_range_start} to {date_range_end}")
    else:
        st.markdown("⚠️ **Note:** Dataset index is not recognized as datetime or is empty. Ensure your dataset's "
                    "index is set to datetime for date range analysis.")

    # Display number of data points per sensor
    data_points_per_sensor = df.notnull().sum()
    st.markdown("🔢 **Number of Data Points per Sensor:**")
    data_points_df = data_points_per_sensor.reset_index().rename(columns={'index': 'Sensor', 0: 'Data Points'})
    with st.expander("Show/Hide Data"):
        st.dataframe(data_points_df)

    # Display any additional general info you think might be helpful
    total_missing_values = df.isnull().sum().sum()
    total_data_points = df.size
    st.markdown(f"**Total Data Points:** {total_data_points}")
    st.markdown(f"**Total Missing Values:** {total_missing_values}")
    if total_missing_values > 0:
        missing_percentage = (total_missing_values / total_data_points) * 100
        st.markdown(f"🚫 **Percentage of Missing Data:** {missing_percentage:.2f}%")

    # If your dataset has more sensors, you might summarize instead of listing each
    num_sensors = len(df.columns)
    st.markdown(f"📊 **Number of Sensors:** {num_sensors}")

    # Optionally, display descriptive statistics
    st.subheader('Descriptive Statistics')
    descriptive_stats = df.describe().transpose()
    with st.expander("Show/Hide Data"):
        st.dataframe(df.describe().style.format("{:.2f}"))


def display_missing_values(df):
    """Calculate and display the percentage of missing values for each sensor."""
    missing_values = df.isnull().mean() * 100
    st.subheader('Missing Values Percentage by Sensor')
    missing_values_df = missing_values.reset_index().rename(columns={'index': 'Sensor', 0: 'Missing Values (%)'})
    with st.expander("Show/Hide Missing Values Data"):
        st.dataframe(missing_values_df)
        # Visualize missing values
    fig_missing = px.bar(missing_values, labels={'value': 'Percentage', 'index': 'Sensor'},
                         title="Missing Values Percentage by Sensor")
    with st.expander("**Show/Hide Missing Values**"):
        st.plotly_chart(fig_missing, use_container_width=True)
    if missing_values.max() > 0:
        st.markdown("🔍 **Observation:** There are missing values in the dataset that may need addressing.")
    else:
        st.markdown("✅ *No* missing values detected in the dataset.")


def check_data_completeness(df, completeness_threshold=90):
    """
    Checks the completeness of the data in the DataFrame.

    :param df: DataFrame to check for completeness.
    :param completeness_threshold: The percentage of completeness considered acceptable for each column.
    """
    # Calculate the percentage of missing values for each column
    missing_percentage = df.isnull().mean() * 100
    total_missing_percentage = df.isnull().mean().mean() * 100

    # Display the percentage of missing data for each column
    # st.write("### Data Completeness Report")

    # Overall data completeness
    st.write(f"#### Overall Data Completeness: {100 - total_missing_percentage:.2f}%")

    # Identify columns below the completeness threshold
    incomplete_columns = missing_percentage[missing_percentage > (100 - completeness_threshold)]
    if not incomplete_columns.empty:
        st.write("### Columns Below Completeness Threshold:")
        for column, percentage in incomplete_columns.iteritems():
            st.write(f"🚫 {column}: {100 - percentage:.2f}% complete (Threshold: {completeness_threshold}%)")
    else:
        st.write(f"✅ All columns are above the {completeness_threshold}% completeness threshold.")


def check_timestamp_consistency(df):
    # Assuming the first column is 'timestamp' and is already in datetime format
    if not df.index.is_monotonic_increasing:
        st.write("🚫 Timestamp inconsistency found: Timestamps are not strictly increasing.")
    else:
        st.write("✅ All timestamps are in order.")


def display_outliers(df):
    """Calculate and display the percentage of outlier values for each sensor using the IQR method."""
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).mean() * 100
    st.subheader('Outliers Percentage by Sensor')
    outliers_df = outliers.reset_index().rename(columns={'index': 'Sensor', 0: 'Outliers (%)'})
    with st.expander("Show/Hide Outliers Data"):
        st.dataframe(outliers_df)
    # Visualize outliers
    fig_outliers = px.bar(outliers, labels={'value': 'Percentage', 'index': 'Sensor'},
                          title="Outliers Percentage by Sensor")
    with st.expander("**Show/Hide Outliers**"):
        st.plotly_chart(fig_outliers, use_container_width=True)
    if outliers.max() > 0:
        st.markdown("🔍 **Observation:** There are outliers in the dataset. Consider reviewing extreme values.")
    else:
        st.markdown("✅ **No** significant outliers detected in the dataset.")


def calculate_missing_values(df):
    """Calculate the percentage of missing values for each sensor."""
    return df.isnull().mean() * 100


def calculate_outliers(df):
    """Calculate the percentage of outlier values for each sensor using the IQR method."""
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    outliers = ((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).mean() * 100
    return outliers


def check_data_accuracy(readings_df, nominal_values_df):
    """
    Checks individual sensor readings against nominal values using vectorized operations,
    and reports the percentage of accuracy issues per sensor, ensuring sensor_ids are in lowercase.

    :param readings_df: DataFrame with sensor readings. Columns are 'timestamp' and each sensor.
    :param nominal_values_df: DataFrame with nominal values for each sensor, including thresholds.
    """
    # Ensure sensor_id is lowercase in nominal_values_df and set as index
    # nominal_values_df.index = nominal_values_df.index.str.lower()

    try:
        nominal_values_df['tag'] = nominal_values_df['tag'].str.lower()
        nominal_values_df.set_index('tag', inplace=True, drop=True)
    except:
        pass
    # Lowercase sensor column names in readings_df, excluding 'timestamp'
    readings_df.columns = [col.lower() if col != 'timestamp' else col for col in readings_df.columns]

    # For each sensor in readings_df, check readings against nominal values
    for sensor_id in readings_df.columns[1:]:  # Exclude 'timestamp' column
        if sensor_id not in nominal_values_df.index:
            # Skip sensors without nominal values
            continue

        sensor_readings = readings_df[sensor_id].dropna()  # Exclude NaNs for accurate percentage calculation
        if sensor_readings.empty:
            st.write(f"🚫 {sensor_id}: No valid data for accuracy assessment.")
            continue

        low_threshold = nominal_values_df.at[sensor_id, 'low_threshold']
        high_threshold = nominal_values_df.at[sensor_id, 'high_threshold']
        threshold_type = nominal_values_df.at[sensor_id, 'threshold_type']

        # Initialize a boolean Series to False
        accuracy_issues = pd.Series([False] * len(sensor_readings), index=sensor_readings.index)

        # Apply vectorized operations for threshold checks
        if threshold_type in ['Down', 'Up/Down']:
            accuracy_issues |= (sensor_readings < float(low_threshold))
        if threshold_type in ['Up', 'Up/Down']:
            accuracy_issues |= (sensor_readings > float(high_threshold))

        # Calculate percentage of accuracy issues
        percentage_issues = accuracy_issues.sum() / len(
            sensor_readings.index) * 100  # mean of boolean Series gives fraction of Trues
        if percentage_issues > 0:
            st.write(f"🔍 **Observation:** Sensor **{sensor_id}** has **{percentage_issues:.2f}%** accuracy issues")


def check_for_duplicates(df):
    total_rows = len(df)
    duplicate_rows_df = df[df.duplicated(keep=False)]  # keep=False marks all duplicates as True
    duplicate_count = len(duplicate_rows_df) / 2  # Since duplicates are counted twice
    percentage_duplicates = (duplicate_count / total_rows) * 100

    if duplicate_count > 0:
        st.write(f"**Duplicate Records Found**: {duplicate_count} ({percentage_duplicates:.2f}%)")
    else:
        st.write("✅ No duplicate records found.")


def visualize_data_quality(df):
    # Calculate metrics
    missing_values = calculate_missing_values(df)
    outliers = calculate_outliers(df)

    # Visualize missing values
    fig_missing = px.bar(missing_values, labels={'value': 'Percentage', 'index': 'Sensor'},
                         title="Missing Values Percentage by Sensor")
    with st.expander("**Show/Hide Missing Values**"):
        st.plotly_chart(fig_missing, use_container_width=True)

    # Visualize outliers
    fig_outliers = px.bar(outliers, labels={'value': 'Percentage', 'index': 'Sensor'},
                          title="Outliers Percentage by Sensor")
    with st.expander("**Show/Hide Outliers**"):
        st.plotly_chart(fig_outliers, use_container_width=True)


def get_plot_title(col, tag_info):
    """ Creates a user-friendly title for the plot based on tag information."""
    if not tag_info.empty:
        title = f"{tag_info['tag_description'].iloc[0]} ({col})"
    else:
        title = col
    return title


def extract_thresholds(tag_info):
    """Extracts low and high thresholds from tag information, handling missing values."""
    low_threshold = tag_info['low_threshold'].iloc[0] if not pd.isna(tag_info['low_threshold'].iloc[0]) else None
    high_threshold = tag_info['high_threshold'].iloc[0] if not pd.isna(tag_info['high_threshold'].iloc[0]) else None
    return low_threshold, high_threshold


# def visualize_sensor_data(df, selected_columns):
#     # Filter the dataframe based on selected columns
#     filtered_data = df[selected_columns]
#
#     # Sensor Summary Statistics
#     st.header("Sensor Summary Statistics")
#     summary_stats = filtered_data.describe()
#     st.write(summary_stats)
#
#     # Time Series Analysis
#     st.header("Time Series Analysis")
#     for col in selected_columns:
#         st.subheader(f"{col}")
#         # print(filtered_data[[col]])
#         st.line_chart(filtered_data[[col]])

def plot_rolling_stats(col, filtered_data, tag_info):
    """
  Plots the rolling statistics with horizontal lines for thresholds (considering threshold_type).
  """
    fig, ax = plt.subplots(figsize=(10, 3))  # Adjust height here
    ax.plot(filtered_data[col], label='Original')
    ax.plot(filtered_data[col].rolling(window=24).mean(), label='1D Rolling Mean')
    # ax.plot(filtered_data[col].rolling(window=24).std(), label='1D Rolling Std')

    low_threshold, high_threshold = extract_thresholds(tag_info)
    threshold_type = tag_info['threshold_type'].iloc[0]

    # Add horizontal lines for thresholds (considering threshold_type)
    if threshold_type == 'Down':
        if low_threshold is not None:
            ax.axhline(y=low_threshold, color='red', linestyle='--', label='Low Threshold')
    elif threshold_type == 'Up/Down':
        if low_threshold is not None:
            ax.axhline(y=low_threshold, color='red', linestyle='--', label='Low Threshold')
        if high_threshold is not None:
            ax.axhline(y=high_threshold, color='green', linestyle='--', label='High Threshold')
    elif threshold_type == 'Up':
        if high_threshold is not None:
            ax.axhline(y=high_threshold, color='green', linestyle='--', label='High Threshold')

    ax.legend()
    plt.xticks(rotation=45)
    st.pyplot(fig)


def visualize_sensor_data(df, selected_columns, tags):
    # Filter the dataframe based on selected columns
    filtered_data = df[selected_columns]

    # Sensor Summary Statistics
    with st.expander("Sensor Summary Statistics", expanded=True):
        summary_stats = filtered_data.describe()
        st.write(summary_stats)

    # Data Quality Assessment (commented out in original code)
    # with st.expander("Data Quality Assessment"):
    #     missing_values = filtered_data.isnull().sum()
    #     st.write("Missing Values per Column")
    #     st.write(missing_values)

    # Detecting Outliers using Z-Score
    with st.expander("Outliers Detection (Z-Score > 3)", expanded=False):
        z_scores = (filtered_data - filtered_data.mean()) / filtered_data.std()
        outliers = z_scores.abs() > 3
        st.write(outliers.sum())

    # Correlation Analysis
    with st.expander("Correlation Analysis", expanded=False):
        correlation_matrix = filtered_data.corr()
        fig, ax = plt.subplots(figsize=(10, 5))  # Adjust height here
        sns.heatmap(correlation_matrix, annot=True, ax=ax, cmap='coolwarm')
        st.pyplot(fig)

    # Time Series Analysis
    with st.expander("Time Series Analysis", expanded=False):
        for col in selected_columns:
            tag_info = tags.loc[tags.tag == col]
            title = get_plot_title(col, tag_info)
            st.write(f'**{title}**')
            plot_rolling_stats(col, filtered_data, tag_info)

    # Histogram and Density Plots
    with st.expander("Histogram and Density Plots", expanded=False):
        for col in selected_columns:
            tag_info = tags.loc[tags.tag == col]
            title = get_plot_title(col, tag_info)
            st.write(f'**{title}**')
            fig, ax = plt.subplots(figsize=(10, 3))  # Adjust height here
            sns.histplot(filtered_data[col], kde=True, ax=ax)
            plt.xticks(rotation=45)
            st.pyplot(fig)

    # Box Plots
    with st.expander("Box Plots", expanded=False):
        fig, ax = plt.subplots(figsize=(10, 3))  # Adjust height here
        sns.boxplot(data=filtered_data, ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    # Trend and Seasonal Decomposition
    with st.expander("Trend and Seasonal Decomposition", expanded=False):
        for col in selected_columns:
            tag_info = tags.loc[tags.tag == col]
            title = get_plot_title(col, tag_info)
            st.write(f'**{title}**')
            result = seasonal_decompose(filtered_data[col], model='additive', period=24)  # Assuming hourly data
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 6))
            result.observed.plot(ax=ax1)
            ax1.set_ylabel('Observed')
            result.trend.plot(ax=ax2)
            ax2.set_ylabel('Trend')
            result.seasonal.plot(ax=ax3)
            ax3.set_ylabel('Seasonal')
            result.resid.plot(ax=ax4)
            ax4.set_ylabel('Residual')
            st.pyplot(fig)

    # Anomaly Detection
    with st.expander("Anomaly Detection", expanded=False):
        for col in selected_columns:
            tag_info = tags.loc[tags.tag == col]
            title = get_plot_title(col, tag_info)
            st.write(f'**{title}**')
            anomalies = detect_anomalies(filtered_data[col])
            fig, ax = plt.subplots(figsize=(10, 3))  # Adjust height here
            ax.plot(filtered_data[col], label='Data')
            ax.scatter(anomalies.index, anomalies, color='red', label='Anomalies')
            plt.xticks(rotation=45)
            ax.legend()
            st.pyplot(fig)


def detect_anomalies(series, threshold=2):
    """Detect anomalies in a series using Z-score method."""
    mean = series.mean()
    std = series.std()
    anomalies = series[(series - mean).abs() > threshold * std]
    return anomalies


#     init_streamlit_comm()
#     df = df.reset_index(names='timestamp')
#
#     # Get an instance of pygwalker's renderer. You should cache this instance to effectively prevent the growth of
#     # in-process memory.
#     @st.cache_resource
#     def get_pyg_renderer() -> "StreamlitRenderer":
#         # When you need to publish your app to the public, you should set the debug parameter to False to prevent
#         # other users from writing to your chart configuration file.
#         return StreamlitRenderer(df, spec="./gw_config.json", debug=False)
#
#     renderer = get_pyg_renderer()
#     return renderer.render_explore()


def calculate_correlation_matrix(df):
    # Calculate the correlation matrix
    corr_matrix = df.corr()
    return corr_matrix


def display_correlation_report(corr_matrix):
    # Format the display of the DataFrame using the Styler.format method
    formatted_corr = corr_matrix.style.background_gradient(cmap='coolwarm').format("{:.2f}")
    st.dataframe(formatted_corr)


def plot_correlation_matrix(corr_matrix):
    st.write("### Correlation Plot")
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=.5,
                cbar_kws={"shrink": .5})
    st.pyplot(plt)


def analyze_strong_correlations(corr_matrix, threshold=0.7):
    """
    Analyzes the correlation matrix to identify pairs of variables with strong correlations.

    :param corr_matrix: The correlation matrix of the variables.
    :param threshold: The threshold above which a correlation is considered strong. Default is 0.7.
    """
    strong_correlations = []

    # Iterate over the correlation matrix
    for i in range(len(corr_matrix.columns)):
        for j in range(i + 1, len(corr_matrix.columns)):  # Avoid self-comparison and redundant pairs
            if abs(corr_matrix.iloc[i, j]) > threshold:
                # Extract the sensor names (or variable names) and the correlation value
                sensor_pair = (corr_matrix.columns[i], corr_matrix.columns[j], corr_matrix.iloc[i, j])
                strong_correlations.append(sensor_pair)

    return strong_correlations


def display_strong_correlations(strong_correlations):
    """
    Displays the pairs of variables with strong correlations in the Streamlit app.

    :param strong_correlations: A list of tuples containing pairs of variables and their correlation coefficient.
    """
    if strong_correlations:
        st.write("### Strong Correlations Detected")
        for pair in strong_correlations:
            sensor_a, sensor_b, correlation = pair
            st.write(f"**{sensor_a}** and **{sensor_b}** have a strong correlation of **{correlation:.2f}**")
    else:
        st.write("No strong correlations detected above the specified threshold.")



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


def visualize_overall_data_quality(missing_readings_df, missing_percentages_df):
    # Check if there is data to plot

    if not missing_readings_df.empty:
        with st.expander("Total Missing Readings per sensor", expanded=False):
            # Plotting the total missing readings per sensor
            plt.figure(figsize=(8, 3))
            missing_totals = missing_readings_df.sum()
            sns.barplot(x=missing_totals.index, y=missing_totals.values)
            plt.title("Total Missing Readings per Sensor")
            plt.xlabel("Sensor")
            plt.ylabel("Total Missing Readings")
            plt.xticks(rotation=90)
            st.pyplot(plt)

    if not missing_percentages_df.empty:
        with st.expander("Total Missing Readings per sensor (%)", expanded=False):

            # Plotting the total missing readings percentages per sensor
            plt.figure(figsize=(8, 3))
            missing_percentage_totals = missing_percentages_df.mean()
            sns.barplot(x=missing_percentage_totals.index, y=missing_percentage_totals.values)
            plt.title("Average Missing Readings Percentage per Sensor")
            plt.xlabel("Sensor")
            plt.ylabel("Average Missing Percentage (%)")
            plt.xticks(rotation=90)
            st.pyplot(plt)


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


def plot_invalid_over_mean(data, invalid_data, sensor):
    plt.figure(figsize=(14, 7))

    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(data.index):
        data.index = pd.to_datetime(data.index)
    if not pd.api.types.is_datetime64_any_dtype(invalid_data['timestamp']):
        invalid_data['timestamp'] = pd.to_datetime(invalid_data['timestamp'])

    # Set timestamp as index for invalid_data
    invalid_data.set_index('timestamp', inplace=True)

    # Plot the mean readings time series
    mean_series = data[f'sum_{sensor}'] / data[f'count_{sensor}']
    plt.plot(mean_series, label='Mean Readings', color='blue')

    # Plot the invalid data points
    invalid_points = invalid_data[invalid_data[f'count_{sensor}_invalid'] > 0]
    plt.scatter(invalid_points.index, mean_series.loc[invalid_points.index], color='red', label='Invalid Readings')

    plt.title(f'Mean Readings and Invalid Readings for {sensor}')
    plt.xlabel('Time')
    plt.ylabel('Mean Reading')
    plt.legend()
    st.pyplot(plt)


def visualize_invalid_data(invalid_readings_df, data):
    sensors = [col.replace('count_', '').replace('_invalid', '') for col in invalid_readings_df.columns if
               '_invalid' in col]

    for sensor in sensors:
        plot_invalid_over_mean(data, invalid_readings_df, sensor)