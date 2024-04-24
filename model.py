from datetime import datetime
import yaml
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import sessionmaker
import pandas as pd

Base = declarative_base()


def load_config(config_path='config-sensor.yaml'):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def construct_sensor_data_class(config):
    attrs = {'__tablename__': config['sensor_data']['table_name']}
    for column in config['sensor_data']['columns']:
        column_type = column['type']
        if column_type == 'Integer':
            attrs[column['name']] = Column(Integer, primary_key=True if column.get('primary_key', False) else False)
        elif column_type == 'String':
            attrs[column['name']] = Column(String)
        elif column_type == 'Float':
            attrs[column['name']] = Column(Float)
        elif column_type == 'DateTime':
            attrs[column['name']] = Column(DateTime)
    return type('SensorData', (Base,), attrs)


def setup_database(config):
    engine = create_engine(config['database_uri'])
    Base.metadata.create_all(engine)

def create_config_file(df, table_name, database_uri="sqlite:///iiot_data.db", file_path='config-sensor.yaml'):
    config = {
        'database_uri': database_uri,
        'sensor_data': {
            'table_name': table_name,
            'columns': []
        }
    }

    # Add timestamp as primary key
    config['sensor_data']['columns'].append({
        'name': 'timestamp',
        'type': 'DateTime',
        'primary_key': True
    })

    # Add other DataFrame columns
    for column in df.columns:
        if column[0].isdigit():
            column = 'sensor_' + column
        config['sensor_data']['columns'].append({
            'name': column,
            'type': 'Float'  # Assuming all columns are Float; adjust if needed
        })

    # Write YAML file
    with open(file_path, 'w') as file:
        yaml.dump(config, file, default_flow_style=False)


# def construct_sensor_data_class(config):
#     attrs = {'__tablename__': config['sensor_data']['table_name'], '__table_args__': {'extend_existing': True}}
#     for column in config['sensor_data']['columns']:
#         column_type = column['type']
#         column_name = column['name']
#         kwargs = {'primary_key': column.get('primary_key', False)}  # Handle primary_key
# 
#         # Prepend 'sensor_' to column names that start with a number
#         if column_name[0].isdigit():
#             column_name = 'sensor_' + column_name
# 
#         # Dynamically assign the column type based on the configuration
#         if column_type == 'Integer':
#             attrs[column_name] = Column(Integer, **kwargs)
#         elif column_type == 'String':
#             attrs[column_name] = Column(String, **kwargs)
#         elif column_type == 'Float':
#             attrs[column_name] = Column(Float, **kwargs)
#         elif column_type == 'DateTime':
#             attrs[column_name] = Column(DateTime, **kwargs)
#         else:
#             raise ValueError(f"Unsupported column type {column_type}")
# 
#     # Create a new class type with all attributes
#     return type('SensorData', (Base,), attrs)


def load_data_to_db(df, session, SensorData):
    for _, row in df.iterrows():
        # Prepare the data for the database insertion
        data_dict = {}
        for column in SensorData.__table__.columns:
            column_name = column.name
            if column_name == 'timestamp':
                # Convert timestamp to datetime if it's not already
                try:
                    data_dict[column_name] = pd.to_datetime(row[column_name]) if not isinstance(row[column_name],
                                                                                                datetime) else row[
                        column_name]
                except KeyError:
                    # Fallback if 'timestamp' is expected to be the DataFrame index
                    data_dict[column_name] = pd.to_datetime(_) if not isinstance(_, datetime) else _
            else:
                data_dict[column_name] = row.get(column_name, None)

        # Create an instance of SensorData using the prepared dictionary
        sensor_data = SensorData(**data_dict)
        session.add(sensor_data)

    try:
        session.commit()
    except Exception as e:
        session.rollback()  # Roll back in case of error during commit
        raise e


def setup_database_engine(database_uri):
    engine = create_engine(database_uri)
    Base.metadata.create_all(engine)  # Create the tables if they don't exist
    return engine


# def setup_database(config):
#     # Setup the database engine
#     engine = create_engine(config['database_uri'])
#     Base.metadata.create_all(engine)  # Make sure all tables are created
#     return engine


def get_session(engine):
    Session = sessionmaker(bind=engine)
    return Session()
