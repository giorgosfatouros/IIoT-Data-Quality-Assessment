import os
import streamlit as st
import pandas as pd
from utils import preprocess_sensor_data, infer_aggregation_frequency
from sqlalchemy.sql import select
from sqlalchemy import Table, MetaData
from sqlalchemy import inspect, create_engine


def get_db_connection(db_user, db_pass, db_ip, db_port, db_name):
    try:
        connection_url = f'leanxcale://{db_user}:{db_pass}@{db_ip}:{db_port}/{db_name}?autocommit=False&parallel=True?txn_mode=NO_CONFLICTS_NO_LOGGING'
        eng = create_engine(connection_url)
        return eng
    except Exception as e:
        st.error(f"Error connecting to the database: {e}")
        return None


def get_table_names(db_connection):
    try:
        if db_connection is None:
            raise ValueError("No database connection available.")
        return inspect(db_connection).get_table_names()
    except Exception as e:
        st.error(f"Error fetching table names: {e}")
        return []


def load_data_from_db(table_name, engine):
    try:
        if engine is None:
            raise ValueError("No database connection available.")

        metadata = MetaData(bind=engine)
        table = Table(table_name, metadata, autoload=True)
        query = select([table])
        with engine.connect() as connection:
            result = connection.execute(query)
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            df.columns = df.columns.str.lower()
            df.set_index('timestamp', inplace=True)
        return df
    except Exception as e:
        st.error(f"Error loading data from table {table_name}: {e}")
        return pd.DataFrame()


# Database connection settings
DB_USER = 'app'
DB_PASS = 'app'
DB_IP = '0.0.0.0'
DB_PORT = '1529'
DB_NAME = 'MOH'

# Read environment variables if needed
DB_USER = os.getenv('DB_USER', DB_USER)
DB_PASS = os.getenv('DB_PASS', DB_PASS)
DB_IP = os.getenv('DB_IP', DB_IP)
DB_PORT = os.getenv('DB_PORT', DB_PORT)
DB_NAME = os.getenv('DB_NAME', DB_NAME)

engine = get_db_connection(DB_USER, DB_PASS, DB_IP, DB_PORT, DB_NAME)


def show():
    st.title('Data Loading, Annotation, and Storage')

    st.write("""
    Use this page to load sensor data from the LeanXscale (LXS) database. You can retrieve aggregated data and perform various analyses on the raw data using these aggregations.
    """)

    table_names = get_table_names(engine)
    table_names = [col for col in table_names if "hours" in col.lower()]

    selected_table = st.sidebar.selectbox(
        "Select a table (machine) to load data from",
        options=table_names,
        placeholder="Choose a machine"
    )

    st.session_state['ORIG_FREQ'] = st.number_input('Set frequency of the original data (in seconds)', min_value=1,
                                                    value=10)

    if selected_table:
        with st.spinner('Loading data...'):
            readings = load_data_from_db(selected_table, engine)
            st.session_state['raw_readings'] = readings
            readings, sensors = preprocess_sensor_data(readings)
            st.session_state['readings'] = readings
            st.session_state['sensors'] = sensors

        inferred_freq = infer_aggregation_frequency(readings)
        if inferred_freq is not None:
            st.success(f"Aggregation frequency: {inferred_freq} seconds")
            st.session_state['AGG_FREQ'] = int(inferred_freq / st.session_state['ORIG_FREQ'])

        else:
            st.warning("Could not infer aggregation frequency from the data")

        with st.expander("Show/Hide Annotated Data"):
            st.dataframe(readings)

        tags_path = "data/tags.csv"
        tags = pd.read_csv(tags_path, header=0)
        tags.columns = tags.columns.str.lower()
        tags.tag = tags.tag.str.lower()
        tags = tags[tags.tag.isin(readings.columns)].reset_index(drop=True)
        tags.columns = [col.replace(' ', '_') for col in tags.columns]

        with st.expander("Show/Hide Data Description"):
            st.dataframe(tags)

        selected_machine = st.selectbox('Select a Machine:', tags['machine_group'].unique())
        st.write(f'You have selected Equipment: **{selected_machine}**')

        machine_tags = tags[
            tags['machine_group'].str.contains(selected_machine[1:], case=False, na=False)
        ].reset_index(drop=True)
        st.session_state['tags'] = machine_tags

        st.write(f'**{machine_tags.shape[0]}** Sensors are monitoring the **{selected_machine}** equipment')

        persist_data = st.checkbox("Export annotated dataset?")
        if persist_data:
            if 'readings' in st.session_state:
                csv = st.session_state['readings'].to_csv().encode('utf-8')
                st.download_button(
                    label="Download data as CSV",
                    data=csv,
                    file_name='annotated_readings.csv',
                    mime='text/csv',
                )
            else:
                st.warning("No sensor data available. Please select data from the side panel.")
        else:
            st.success(
                "Data successfully prepared for visualization. Please proceed to the **Visualization** page to "
                "explore your data.")
    else:
        st.warning("Please select a Machine for analysis.")
