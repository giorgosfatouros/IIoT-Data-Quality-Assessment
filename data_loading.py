import streamlit as st
import pandas as pd
from utils import preprocess_sensor_data
from sqlalchemy.sql import select
from sqlalchemy import Table, MetaData
from sqlalchemy import inspect
from sqlalchemy import create_engine
from model import get_session


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


# Connection
DB_USER = 'app'
DB_PASS = 'app'
DB_IP = '0.0.0.0'
DB_PORT = '1529'
DB_NAME = 'MOH'

engine = get_db_connection(DB_USER, DB_PASS, DB_IP, DB_PORT, DB_NAME)


def show():
    st.title('Data Loading and Storage')

    # uploaded_file2 = st.file_uploader("**Upload Sensor and Equipment/Machine descriptions**", type='csv', key='tags')
    # if uploaded_file2 is not None:

    table_names = get_table_names(engine)
    table_names = [col for col in table_names if "hours" in col.lower()]

    selected_table = st.sidebar.selectbox(
        "Select a table (machine) to load data from",
        options=table_names,
        placeholder="Choose a machine"
    )

    if selected_table:
        with st.spinner('Loading data...'):
            readings = load_data_from_db(selected_table, engine)
            st.session_state['raw_readings'] = readings
            readings, sensors = preprocess_sensor_data(readings)
            st.session_state['readings'] = readings
            st.session_state['sensors'] = sensors

        with st.expander("Show/Hide Data"):
            st.dataframe(readings)

        uploaded_file2 = "data/tags.csv"
        tags = pd.read_csv(uploaded_file2, header=0)
        tags.columns = tags.columns.str.lower()
        tags.tag = tags.tag.str.lower()
        tags = tags[tags.tag.isin(readings.columns)].reset_index(drop=True)


        # tags = tags.iloc[:, 1:]
        tags.columns = [col.replace(' ', '_') for col in tags.columns]
        with st.expander("Show/Hide Data Description"):
            st.dataframe(tags)
        selected_machine = st.selectbox('Select a Machine:', tags['machine_group'].unique())
        st.write(f'You have selected Equipment: **{selected_machine}**')
        machine_tags = tags[
            tags['machine_group'].str.contains(selected_machine[1:], case=False, na=False)].reset_index(
            drop=True)
        st.session_state['tags'] = machine_tags

        st.write(f'**{machine_tags.shape[0]}** Sensors are monitoring the **{selected_machine}** equipment')

        # persist_data = st.checkbox("Persist data in the database for overall analytics?")
        persist_data = False
        if persist_data:
            session = get_session(engine)
            load_data_to_db(readings, session)
            load_data_to_db(readings, session, SensorData)

            st.success("Data successfully saved to the database!")
            st.success("Data successfully prepared for visualization. Please proceed to the **Visualization** "
                       "page to explore your data.")
        else:
            st.success("Data successfully prepared for visualization. Please proceed to the **Visualization** "
                       "page to explore your data.")
    else:
        st.warning("Please select at Machine for analysis.")
