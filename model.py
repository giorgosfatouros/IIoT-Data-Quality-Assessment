import yaml
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime
from sqlalchemy.orm import sessionmaker

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
    return engine


def get_session(engine):
    Session = sessionmaker(bind=engine)
    return Session()



