# IIoT Data Quality Assessment App

This project is a Streamlit-based application designed to analyze and assess the quality of data collected from Industrial Internet of Things (IIoT) devices. 
## Dependencies 
This app is design to read data from [LeanXcale database](https://www.leanxcale.com/real-time-analytics), thus the data used for analysis should be stored in this database.

## Features 
- **Data Loading**: Import raw data or connect to the LeanXcale (LXS) database via Kafka for real-time data streaming. 
- **LeanXscale Integration**: Leverage online aggregates and incremental analytics for fast and efficient data processing. 
- **Data Annotation**: The aggregated data are automatically annotated based on the nominal sensor values and can be exported to for further exploitation. 
- **Data Visualization**: Visualize the loaded and aggregated data to understand its structure and quality. 
- **Missing Values Analysis**: Detect and handle missing values in the raw sensor dataset, utilizing aggregated data. 
- **Invalid Values Analysis**: Identify and analyze invalid readings or alarms from your sensors. 
- **Data Quality**: Perform comprehensive data quality assessments, including metrics and visualizations.

## Setup

### Prerequisites
- Python 3.8 
- Docker
- LeanXscale docker service

### Step 1: Clone the Project
```bash
git clone https://github.com/giorgosfatouros/IIoT-Data-Quality-Assessment.git
cd iiot-data-quality-assessment-app
```

### Step 2: Start LeanXcale docker service (if needed)
```bash
docker run --name leanxcaledb-service --env KVPEXTERNALIP='leanxcaledb-service!9800' -p 0.0.0.0:1529:1529 -d ferrari 
```
For Installing LeanXscale refer here: https://gitlab.gftinnovation.eu/fame/leanxcaledb.git

### Local Installation

1. Navigate to the project directory:

```bash
cd iiot-data-quality-assessment-app

```
2. Create and activate a virtual environment:
```bash
python -m venv iot
source iot/bin/activate
```

3. Install the LeanXcale Python client and project dependencies:
```bash
pip install pyLeanxcale-1.9.13_latest-py3-none-any.whl 
```
4. Install any additional requirements:
```bash
pip install requirements.txt
```
#### Step 4: Running the App
To run the app, navigate to the project directory in your terminal and execute:
```bash
streamlit run app.py
```
### Docker Installation 

```bash
docker-compose up -d
```

### 4. Usage 
Go to: http://www.localhost:8501
Follow the instructions within the app to upload your sensor data.
 1. **Data Loading**: Navigate to the **Data Loading** page to select the data table (machine) for analysis.
 2. **Data Visualization**: Use the **Data Visualization** page to explore the data through various visualizations.
 3. **Missing Values Analysis**: Go to the **Missing Values Analysis** to get insights for missing values into the original/raw data.
 4. **Invalid Values Analysis**: The **Invalid Values Analysis** page helps you identify and understand invalid readings from your sensor data.
 5. **Data Quality**: Access the **Data Quality** page for a detailed assessment of your data's quality, including completeness, accuracy, and consistency.

### Acknowledgements
The project has received funding from the European Union’s funded Project HEU FAME under Grant Agreement No. 101092639.

