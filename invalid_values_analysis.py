import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt


def plot_invalid_over_mean(data, sensor, threshold):
    """
    Plot the mean readings time series and highlight invalid readings.
    """
    mean_series = data[sensor]

    with st.expander(f'Mean Readings and Invalid Readings for {sensor}', expanded=False):
        plt.figure(figsize=(10, 3))
        plt.plot(mean_series, label='Mean Readings', color='blue')

        invalid_points = data[data[f'{sensor}_alarms'] >= threshold]
        plt.scatter(invalid_points.index, mean_series.loc[invalid_points.index], color='red', label='Invalid Readings')

        plt.xlabel('Time')
        plt.ylabel('Mean Reading')
        plt.legend()
        st.pyplot(plt)


def visualize_invalid_data(data, threshold):
    """
    Visualize the invalid data for all sensors based on the threshold.
    """
    sensors = [col for col in data.columns if not 'alarm' in col in col]

    for sensor in sensors:
        plot_invalid_over_mean(data, sensor, threshold)


def display_overall_statistics(readings, expected_readings_per_hour):
    """
    Display overall statistics for the sensor readings focusing on sensors with at least one alarm.
    """
    alarm_columns = [col for col in readings.columns if '_alarms' in col]

    # Filter columns with at least one alarm
    alarms_df = readings[alarm_columns]
    sensors_with_alarms = alarms_df.loc[:, (alarms_df > 0).any(axis=0)]

    if sensors_with_alarms.empty:
        st.write("No alarms detected in any sensor.")
        return

    total_readings = readings.shape[0] * expected_readings_per_hour
    total_alarms = sensors_with_alarms.sum().sum()
    avg_alarms_per_sensor = sensors_with_alarms.mean().mean()
    max_alarms_sensor = sensors_with_alarms.sum().idxmax()
    max_alarms_count = sensors_with_alarms.sum().max()

    with st.expander(f'Overall Statistics of Invalid/Alarm Data', expanded=False):
        st.write(f"**Total Readings:** {total_readings}")
        st.write(f"**Total Alarms:** {total_alarms}")
        st.write(f"**Average Alarms per Sensor:** {avg_alarms_per_sensor:.2f}")
        st.write(f"**Sensor with Most Alarms:** {max_alarms_sensor} ({max_alarms_count} alarms)")

    with st.expander(f'Detailed Report of Sensors with Alarms', expanded=False):
        for sensor in sensors_with_alarms.columns:
            total_sensor_alarms = sensors_with_alarms[sensor].sum()
            st.write(f"**{sensor}:** {total_sensor_alarms}")


def update_threshold():
    st.session_state.threshold_changed = True


def show():
    st.title('Invalid Values Analysis')
    st.write("This page analyzes the invalid readings or alarms from sensor data, which are aggregated "
             "from the original frequency for fast and efficient analysis. The focuses on detecting and understanding "
             "alarms triggered by invalid readings in the data.")

    if 'threshold_changed' not in st.session_state:
        st.session_state.threshold_changed = False

    threshold = st.slider('Set threshold for invalid readings',
                          min_value=1,
                          max_value=st.session_state['AGG_FREQ'],
                          value=1,
                          key='threshold',
                          on_change=update_threshold)

    if 'tags' in st.session_state:
        tags = st.session_state['tags']
        with st.expander("Show/Hide Data Description"):
            st.dataframe(tags)

    if 'readings' in st.session_state:
        readings = st.session_state['readings']
        # Show overall statistics
        display_overall_statistics(readings, st.session_state['AGG_FREQ'])
        # Visualization
        visualize_invalid_data(readings, threshold)
    else:
        st.warning("No sensor data available. Please upload and process data in the Data Loading page.")

    if st.session_state.threshold_changed:
        st.session_state.threshold_changed = False
        st.rerun()


if __name__ == "__main__":
    show()
