# IIoT Data Quality Assessment App

This project is a Streamlit-based application designed for assessing the data quality of IIoT (Industrial Internet of Things) sensor data. It allows users to upload sensor data, optionally persist it in a database for analytics, and assess data quality metrics.

## Setup

Before running the app, you need to set up a configuration file and a database. Follow these steps to set up your project:

### 1. Install Dependencies

Ensure you have Python and [Poetry](https://python-poetry.org/docs/#installation) installed on your system. Poetry handles dependency management and virtual environments. Follow the steps below to set up your project dependencies:

1. **Install Poetry**: If you haven't installed Poetry yet, follow the [official installation guide](https://python-poetry.org/docs/#installation).

2. **Set Up the Project**: Navigate to the project directory in your terminal.

3. **Install Dependencies**: Run the following command to install the project dependencies specified in the `pyproject.toml` file:



```bash
pip install pyLeanxcale-1.9.13_latest-py3-none-any.whl 

poetry install

```

### 2. Configuration File
Create a `config-sensor.yaml` file in the root directory of your project. This file will describe the structure of the data to be uploaded. Here is an example configuration:
``` 
database_uri: "sqlite:///iiot_data.db"
sensor_data:
  table_name: "sensor_data"
  columns:
    - name: "timestamp"
      type: "DateTime"
      format: "%Y-%m-%d %H:%M:%S"
    - name: "sensor_id"
      type: "String"
    - name: "value"
      type: "Float"
```

Adjust the `database_uri` and the structure under `sensor_data` to match your requirements. The type can be one of Integer, String, Float, or DateTime. If DateTime is used, specify the format to correctly parse the dates.

### 3. Running the App

First start if stopped the LeanXscale Datastore
```bash
docker restart leanxcaledb-service
```
To run the app, navigate to the project directory in your terminal and run:

```bash
streamlit run app.py
```

### 4. Uploading Data
Follow the instructions within the app to upload your sensor data. The configuration in `config-sensor.yaml` will determine how the data is processed and stored.

### 5. Extending the App
To add new features or modify the app, refer to the Streamlit documentation and the SQLAlchemy ORM documentation for guidance. The modular design makes it easy to extend and customize for your specific needs.

### Acknowledgements
The project has received funding from the European Union’s funded Project HEU FAME under Grant Agreement No. 101092639.

