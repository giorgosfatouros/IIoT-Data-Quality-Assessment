# IIoT Data Quality Assessment Service

A full-stack web application designed to analyze and assess the quality of high frequency data collected from Industrial Internet of Things (IIoT) sensors, efficiently. 

This application consists of:
- **FastAPI Backend**: RESTful API for data processing and analytics
- **React Frontend**: Modern, responsive dashboard interface
- **TimescaleDB Integration**: PostgreSQL with TimescaleDB extension for time-series data optimization
- **DQA Worker Service**: Background service for data aggregation and quality assessment

## Features 
- **Modern Web Interface**: React-based responsive dashboard with professional UI/UX
- **Data Import API**: Upload and import CSV sensor data files via RESTful API
  - File validation and preview
  - Automatic machine type detection
  - Background processing with progress tracking
  - Direct import to TimescaleDB raw_sensor_data table
- **Data Loading**: Interactive data source selection and preprocessing with real-time previews
- **TimescaleDB Integration**: Leverage hypertables, compression, and continuous aggregates for efficient time-series data processing
- **Advanced Analytics**: Comprehensive visualization analytics including:
  - Summary statistics and correlation analysis
  - Time series analysis with trend detection
  - Histogram and density plots
  - Box plots and seasonal decomposition
  - Anomaly detection using statistical methods
- **Data Visualization**: Interactive charts and graphs using modern charting libraries
- **Missing Values Analysis**: Detect and handle missing values in the raw sensor dataset
- **Invalid Values Analysis**: Identify and analyze invalid readings or alarms from sensors
- **Data Quality Assessment**: Comprehensive data quality metrics and visualizations
- **RESTful API**: Well-documented FastAPI backend with automatic OpenAPI documentation

## Architecture

```
┌─────────────────┐    HTTP/REST     ┌──────────────────┐    SQLAlchemy    ┌─────────────────┐
│   React Frontend│ ◄──────────────► │  FastAPI Backend │ ◄──────────────► │  TimescaleDB    │
│                 │                  │   (Port 8000)    │                  │  (Port 5432)    │
└─────────────────┘                  └──────────────────┘                  └─────────────────┘
                                             │ ▲                                    │ ▲
                                             │ │                                    │ │
                                             │ │                                    │ │
                                             ▼ │                                    ▼ │
                                     ┌──────────────────┐                  ┌──────────────────┐
                                     │  DQA Worker      │                  │  raw_sensor_data │
                                     │  (Background)    │                  │  aggr_insights   │
                                     └──────────────────┘                  └──────────────────┘
```

## Prerequisites
- **Python 3.9+** for backend development
- **Node.js 18+** for frontend development  
- **Docker** and **Docker Compose** for containerized services
- **TimescaleDB** (PostgreSQL with TimescaleDB extension) - automatically set up via Docker Compose
  - [TimescaleDB Documentation](https://docs.timescale.com/)
  - [Hypertables and Compression](https://docs.timescale.com/use-timescale/latest/hypertables/)
- **Sensor Metadata** in CSV format (see Data Description Requirements below)


### Data Description Requirements

Provide a `csv` file describing the sensors. The file should adhere to the following format and contains the specified columns. Below is a detailed description of each required column and its expected content:

### TAG:
- **Description**: A unique identifier for each sensor.
- **Example**: `22PI102`

### Tag Description:
- **Description**: A brief description of the tag or sensor, explaining what it measures or monitors.
- **Example**: `SEAL OIL MAIN PUMP PRESSURE`

### MACHINE_GROUP:
- **Description**: The group or category of machinery to which the tag belongs.
- **Example**: `K-2201`

### LOW_THRESHOLD:
- **Description**: The lower limit or threshold for the acceptable range of the tag's measurements.
- **Example**: `6`

### HIGH_THRESHOLD:
- **Description**: The upper limit or threshold for the acceptable range of the tag's measurements. If not applicable, it can be left empty.
- **Example**: `NaN`

### THRESHOLD_TYPE:
- **Description**: Indicates the type of threshold (e.g., "Down" for lower limit thresholds). One of (`Up, Down or Up/Down`)
- **Example**: `Down`

### AGGREGATION_RULE:
- **Description**: The rule for aggregating data points (e.g., `min` for minimum value).
- **Example**: `min`

### ENGINEERING_UNITS:
- **Description**: The units in which the measurements are recorded.
- **Example**: `Kgf/cm2`

### CATEGORY:
- **Description**: The category of the measurement, such as `Pressure`, `Temperature`, etc.
- **Example**: `Pressure`

### Sample Data Format

| TAG     | Tag Description                         | MACHINE_GROUP | LOW_THRESHOLD | HIGH_THRESHOLD | THRESHOLD_TYPE | AGGREGATION_RULE | ENGINEERING_UNITS | CATEGORY |
|---------|-----------------------------------------|---------------|---------------|----------------|----------------|------------------|-------------------|----------|
| 22PI102 | SEAL OIL MAIN PUMP PRESSURE             | K-2201 | 6             | NaN            | Down           | min              | Kgf/cm2           | Pressure |
| 22PI103 | CONTROL OIL HEADER PRESSURE             | K-2201 | 5             | NaN            | Down

## Installation 
### Clone the Project
```bash
git clone https://github.com/giorgosfatouros/IIoT-Data-Quality-Assessment.git
cd iiot-data-quality-assessment-app
```

### Quick Start with Docker Compose

1. Clone the repository and navigate to the project directory:
```bash
git clone <repository-url>
cd fame-data-quality-assessment
```

2. Create `.env` file from example:
```bash
cp env.example .env
# Edit .env and add your OPENAI_API_KEY
```

3. Start all services:
```bash
./start-dev.sh
```

This will start:
- TimescaleDB database (port 5432)
- DQA Worker service (background aggregation)
- FastAPI Backend (port 8000)
- React Frontend (port 5173)

### Manual Installation

1. Navigate to the project directory:
```bash
cd fame-data-quality-assessment
```

2. Set up backend:
```bash
cd backend
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

3. Set up frontend:
```bash
cd frontend
npm install
```

4. Start TimescaleDB with Docker:
```bash
docker-compose up -d timescaledb
```

5. Start backend:
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

6. Start frontend (in another terminal):
```bash
cd frontend
npm run dev
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

## Data Import API

The application includes a RESTful API for uploading and importing sensor data CSV files.

### Setup

The Data Import API is part of the FastAPI backend service. All services, including the DQA Worker for background data aggregation, are automatically set up when you start the application using Docker Compose:

```bash
./start-dev.sh
```

This will start:
- TimescaleDB database
- DQA Worker service (handles background data aggregation)
- FastAPI Backend (includes the Data Import API)
- React Frontend

For manual setup, ensure the backend dependencies are installed using `uv` (the project uses `pyproject.toml` for dependency management):
```bash
cd backend
uv pip install -e .
```

### Quick Usage

**Upload and import data:**
```bash
curl -X POST "http://localhost:8000/import/upload" \
  -F "data_file=@/path/to/sensor_data.csv" \
  -F "tags_file=@/path/to/tags.csv" \
  -F "table_name=KT2201" \
  -F "machine_type=AUTO"
```

**Check import status:**
```bash
curl "http://localhost:8000/import/status/{job_id}"
```

**Validate files before import:**
```bash
curl -X POST "http://localhost:8000/import/validate" \
  -F "data_file=@/path/to/sensor_data.csv" \
  -F "tags_file=@/path/to/tags.csv"
```

**Test the API:**
```bash
cd backend
python test_import_api.py
```

For complete API documentation, see **[DATA_IMPORT_API.md](DATA_IMPORT_API.md)**.

### Supported Machine Types
- **KT2201**: K-2201/KT-2201 Machine
- **K3301**: K-3301/KT-3301 Machine  
- **K5700**: K-5700 Machine
- **AUTO**: Automatic detection from filename or column patterns

## Citation

If you use this software in your research, please cite:

```bibtex
@inproceedings{fatouros2023comprehensive,
  title={Comprehensive architecture for data quality assessment in industrial iot},
  author={Fatouros, Georgios and Makridis, Georgios and Mavrogiorgou, Argyro and Soldatos, John and Filippakis, Michael and Kyriazis, Dimosthenis},
  booktitle={2023 19th International Conference on Distributed Computing in Smart Systems and the Internet of Things (DCOSS-IoT)},
  pages={512--517},
  year={2023},
  organization={IEEE}
}
```

## Acknowledgements

The project has received funding from the European Union's funded **Project HEU FAME** under Grant Agreement No. **101092639**.

