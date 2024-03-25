import pandas as pd
import plotly.express as px
import streamlit as st
from pygwalker.api.streamlit import StreamlitRenderer, init_streamlit_comm
import seaborn as sns
import matplotlib.pyplot as plt

def preprocess_sensor_data(df):
    # drop empty rows/cols
    df = df.iloc[2:, 4:].reset_index(drop=True)
    # Set the column names to the values in the first row
    df.columns = df.iloc[0]
    # Drop the first row as it's now the header
    df = df.drop(df.index[0])
    df = df.reset_index(drop=True)
    # rename cols
    df.columns = [col.replace(' - SnapShot', '') for col in df.columns]
    df.columns = [col.lower().replace(' ', '_') for col in df.columns]
    df.columns = [col.replace('.pv', '') for col in df.columns]
    try:
        df = df.set_index('timestamp')
        df.index = pd.to_datetime(df.index)
    except Exception as e:
        print(f'Exception at preprocess_sensor_data():\n{e}')
        pass
    for col in df.columns:
        # Convert only if the column type is 'object'
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


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
            accuracy_issues |= (sensor_readings < low_threshold)
        if threshold_type in ['Up', 'Up/Down']:
            accuracy_issues |= (sensor_readings > high_threshold)

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


def visualize_sensor_data(df):
    init_streamlit_comm()
    df = df.reset_index(names='timestamp')

    # Get an instance of pygwalker's renderer. You should cache this instance to effectively prevent the growth of
    # in-process memory.
    @st.cache_resource
    def get_pyg_renderer() -> "StreamlitRenderer":
        # When you need to publish your app to the public, you should set the debug parameter to False to prevent
        # other users from writing to your chart configuration file.
        return StreamlitRenderer(df, spec="./gw_config.json", debug=False)

    renderer = get_pyg_renderer()
    return renderer.render_explore()


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
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, linewidths=.5, cbar_kws={"shrink": .5})
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
        for j in range(i+1, len(corr_matrix.columns)):  # Avoid self-comparison and redundant pairs
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
