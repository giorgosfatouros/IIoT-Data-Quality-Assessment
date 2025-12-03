# Getting Started Guide

This guide will help you set up and start using the IIoT Data Quality Assessment Service for the first time.

## Prerequisites

Before you begin, ensure you have the following installed on your system:

- **Docker** and **Docker Compose** - Required for running the application services
- **Node.js 18+** - Required for running the frontend development server
- **Git** - For cloning the repository (if needed)

> **Note**: Python is not required for end users as the backend runs in Docker containers.

## Quick Start

### Step 1: Clone the Repository

If you haven't already, clone the repository:

```bash
git clone <repository-url>
cd fame-data-quality-assessment
```

### Step 2: Create Environment Configuration

Create a `.env` file from the example template:

```bash
cp env.example .env
```

### Step 3: Configure Environment Variables

Edit the `.env` file and add your OpenAI API key (required for the DQA Agent feature):

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

> **Note**: The DQA Agent feature requires a valid OpenAI API key. Other features will work without it.

### Step 4: Start the Application

Run the startup script:

```bash
./start-dev.sh
```

This script will:
- Check that Docker is running
- Create the `.env` file if it doesn't exist
- Build and start all required services:
  - TimescaleDB database
  - DQA Worker service (background data aggregation)
  - FastAPI Backend API
- Wait for all services to be ready
- Start the frontend development server

### Step 5: Access the Application

Once all services are running, you can access:

- **Frontend Application**: http://localhost:5173
- **Backend API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

## First Steps

### 1. Verify Services are Running

After starting the application, you should see a confirmation message indicating all services are ready. If any service fails to start, check the troubleshooting guide.

### 2. Navigate to the Home Page

Open your web browser and go to http://localhost:5173. You should see the home page with cards for each feature:

- **Data Loading** - Select and view sensor data
- **Data Visualization** - Explore data through charts
- **Missing Values** - Analyze missing data
- **Invalid Values** - Detect invalid readings
- **Data Quality** - Comprehensive quality assessment
- **DQA Agent** - AI-powered chat assistant

### 3. Import Your First Dataset

Before you can analyze data, you need to import sensor data:

1. Navigate to **Data Loading** from the home page
2. Click on the **Upload Data** tab
3. Follow the instructions in the [Data Import Guide](data-import-guide.md)

### 4. Explore Your Data

Once data is imported:

1. Select a machine group from the dropdown in **Data Loading**
2. View raw or preprocessed sensor data
3. Navigate to **Data Visualization** to see charts and analytics
4. Check **Data Quality** for comprehensive quality metrics

## Application Structure

The application consists of several interconnected services:

### Frontend (React)
- Modern web interface running on port 5173
- Provides all user interactions and visualizations
- Hot-reload enabled for development

### Backend (FastAPI)
- RESTful API running on port 8000
- Handles data processing, analytics, and import operations
- Provides API documentation at `/docs`

### Database (TimescaleDB)
- PostgreSQL with TimescaleDB extension
- Stores raw sensor data and aggregated insights
- Optimized for time-series data

### Worker Service
- Background service for data aggregation
- Automatically processes and aggregates sensor data
- Runs continuously in the background

## Stopping the Application

To stop all services:

```bash
./stop-dev.sh
```

Or manually:

```bash
docker-compose down
```

## Next Steps

Now that you have the application running:

1. **Import Data**: Follow the [Data Import Guide](data-import-guide.md) to upload your sensor data
2. **Load Data**: Use the [Data Loading Guide](data-loading-guide.md) to select and view your data
3. **Visualize**: Explore the [Data Visualization Guide](data-visualization-guide.md) for charts and analytics
4. **Assess Quality**: Check the [Data Quality Guide](data-quality-guide.md) for comprehensive assessment

## Troubleshooting

If you encounter issues during setup:

- Check the [Troubleshooting Guide](troubleshooting.md) for common problems
- Verify Docker is running: `docker info`
- Check service logs: `docker-compose logs`
- Ensure ports 5173, 8000, and 5432 are not in use by other applications

## Related Documentation

- [Data Import Guide](data-import-guide.md) - Upload and import sensor data
- [Data Loading Guide](data-loading-guide.md) - Select and view data
- [Main Documentation Index](README.md) - Overview of all documentation

---

*For technical details and development information, see the main [README](../README.md).*

