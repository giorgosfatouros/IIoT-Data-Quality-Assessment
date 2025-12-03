# Frequently Asked Questions (FAQ)

Common questions and answers about using the IIoT Data Quality Assessment Service.

## General Questions

### What is the IIoT Data Quality Assessment Service?

The IIoT Data Quality Assessment Service is a web application for analyzing and assessing the quality of high-frequency sensor data from Industrial Internet of Things (IIoT) devices. It provides tools for data import, visualization, quality assessment, and AI-powered insights.

### What are the main features?

- **Data Import**: Upload and import CSV sensor data files
- **Data Visualization**: Interactive charts and advanced analytics
- **Missing Values Analysis**: Identify and analyze missing data
- **Invalid Values Analysis**: Detect threshold violations
- **Data Quality Assessment**: Comprehensive quality metrics
- **DQA Agent**: AI-powered chat assistant

### What browsers are supported?

The application works best with:
- Chrome/Edge (latest versions)
- Firefox (latest versions)
- Safari (latest versions)

### Do I need to install anything?

For end users, you only need:
- Docker and Docker Compose (for services)
- Node.js 18+ (for frontend)
- A web browser

The backend runs in Docker containers, so Python is not required for end users.

## Installation and Setup

### How do I install the application?

See the [Getting Started Guide](getting-started.md) for detailed installation instructions. The quickest way is:

1. Clone the repository
2. Create `.env` file from `env.example`
3. Add your OpenAI API key (for DQA Agent)
4. Run `./start-dev.sh`

### What is the OpenAI API key for?

The OpenAI API key is required for the DQA Agent feature, which provides AI-powered chat assistance. Other features work without it.

### How do I start the application?

Run the startup script:
```bash
./start-dev.sh
```

This starts all services automatically.

### How do I stop the application?

Run the stop script:
```bash
./stop-dev.sh
```

Or manually:
```bash
docker-compose down
```

## Data Import

### What file formats are supported?

The application supports CSV files for:
- Sensor data (with timestamp column)
- Sensor metadata/tags (with required columns)

See [Data Import Guide](data-import-guide.md) for detailed format requirements.

### What columns are required in the data file?

The data file must have:
- A `timestamp` column (first column)
- One column per sensor tag

### What columns are required in the tags file?

The tags file must have these columns:
- TAG
- Tag Description
- MACHINE_GROUP
- LOW_THRESHOLD
- HIGH_THRESHOLD
- THRESHOLD_TYPE
- AGGREGATION_RULE
- ENGINEERING_UNITS
- CATEGORY

### Can I import only specific sensors?

Yes, you can specify a comma-separated list of sensor tags during import. Leave empty to import all sensors.

### How long does import take?

Import time depends on data size:
- Small datasets (< 10,000 rows): 1-2 minutes
- Medium datasets (10,000 - 100,000 rows): 5-10 minutes
- Large datasets (> 100,000 rows): 10+ minutes

### What if import fails?

Check the [Troubleshooting Guide](troubleshooting.md) for import issues. Common causes:
- File format errors
- Database connection issues
- Insufficient disk space

## Data Analysis

### How do I select data for analysis?

1. Go to **Data Loading** page
2. Select a machine group from the dropdown
3. Data loads automatically

See [Data Loading Guide](data-loading-guide.md) for details.

### What is the difference between raw and preprocessed data?

- **Raw Data**: Original sensor readings as imported
- **Preprocessed Data**: Aggregated data based on aggregation rules (min, max, avg, sum)

### How do I filter data by date range?

Many pages support date range filtering:
1. Set **Date From** and **Date To**
2. Click **Apply Filter**
3. Analysis updates for selected range

### Can I export data?

Export options vary by page:
- Screenshots: Use browser tools
- Data export: Available on some pages (CSV format)
- Reports: Generate quality reports

## Data Quality

### What is data quality?

Data quality refers to the fitness of data for its intended use, measured by:
- **Completeness**: How much data is present
- **Accuracy**: How correct the data is
- **Consistency**: How uniform the data is

### How is data quality assessed?

The system evaluates:
- Missing value percentages
- Threshold violations (accuracy)
- Duplicate records (consistency)
- Outlier detection
- Correlation analysis

See [Data Quality Guide](data-quality-guide.md) for details.

### What is a good data quality score?

Quality levels:
- **Excellent**: All metrics within acceptable ranges
- **Good**: Minor issues, generally acceptable
- **Fair**: Some concerns, monitor closely
- **Poor**: Significant issues, take action

### What should I do if quality is poor?

1. Identify specific issues (missing values, accuracy, etc.)
2. Investigate root causes
3. Address sensor hardware issues
4. Check data collection system
5. Re-assess after improvements

## Missing Values

### What are missing values?

Missing values are expected sensor readings that are not recorded. They can occur due to:
- Sensor malfunctions
- Data collection interruptions
- Transmission failures

### How are missing values detected?

The system compares expected readings (based on timestamp frequency) with actual readings to identify gaps.

### What is an acceptable missing value percentage?

Guidelines:
- **< 5%**: Excellent
- **5-10%**: Good
- **10-20%**: Moderate concerns
- **> 20%**: Poor, take action

### How do I fix missing values?

1. Investigate sensor hardware
2. Check data collection system
3. Review transmission logs
4. Consider sensor replacement if needed

## Invalid Values

### What are invalid values?

Invalid values are sensor readings that violate defined thresholds:
- Below low threshold (for "Down" type)
- Above high threshold (for "Up" type)
- Outside range (for "Up/Down" type)

### How are thresholds defined?

Thresholds are defined in the tags metadata file:
- LOW_THRESHOLD: Lower limit
- HIGH_THRESHOLD: Upper limit
- THRESHOLD_TYPE: Up, Down, or Up/Down

### What is an acceptable alarm rate?

Guidelines:
- **< 5%**: Good data quality
- **5-15%**: Some concerns, investigate
- **> 15%**: Significant issues, take action

### How do I fix invalid values?

1. Check sensor calibration
2. Verify threshold settings
3. Review sensor hardware
4. Check operating conditions

## DQA Agent

### What is the DQA Agent?

The DQA Agent is an AI-powered chat assistant that helps you:
- Ask questions about data quality
- Get insights and recommendations
- Understand quality metrics
- Troubleshoot issues

### Do I need an OpenAI API key?

Yes, the DQA Agent requires a valid OpenAI API key. Other features work without it.

### What types of questions can I ask?

You can ask about:
- Data quality metrics
- Specific sensors
- Recommendations
- Metric explanations
- Troubleshooting help

See [DQA Agent Guide](dqa-agent-guide.md) for examples.

### Are agent responses always accurate?

Agent responses are AI-generated and should be verified. Use as guidance, not absolute truth. Cross-check important findings with other analyses.

## Troubleshooting

### Services won't start

1. Check Docker is running: `docker info`
2. Verify ports are not in use
3. Check `.env` file exists and is configured
4. Review startup logs: `docker-compose logs`

### Frontend not loading

1. Check if frontend is running: `lsof -ti tcp:5173`
2. Verify Node.js is installed: `node --version`
3. Check for port conflicts
4. Review browser console for errors

### Backend API errors

1. Check backend logs: `docker-compose logs backend`
2. Verify database connection
3. Check backend health: `curl http://localhost:8000/health`
4. Restart backend: `docker-compose restart backend`

### Database connection issues

1. Check TimescaleDB is running: `docker-compose ps timescaledb`
2. Verify environment variables
3. Check database logs: `docker-compose logs timescaledb`
4. Test connection: `docker-compose exec timescaledb pg_isready`

See [Troubleshooting Guide](troubleshooting.md) for more solutions.

## Performance

### Application is slow

1. Reduce date range in filters
2. Select fewer sensors
3. Use aggregated data source
4. Check system resources
5. Close unnecessary browser tabs

### Charts take long to render

1. Use date range filters
2. Select fewer sensors
3. Wait for charts to fully load
4. Check data volume

### Import is slow

Large files take time. Monitor progress and be patient. Consider importing subset of sensors first.

## Best Practices

### Data Import

1. Always validate files before importing
2. Start with small test dataset
3. Backup existing data before overwriting
4. Verify file encoding (UTF-8)
5. Check sensor tags match between files

### Data Analysis

1. Start with overview (all sensors)
2. Focus on specific sensors if needed
3. Use date filters for large datasets
4. Compare results across time periods
5. Document findings

### Data Quality

1. Assess quality regularly
2. Set quality standards
3. Monitor trends over time
4. Take action on identified issues
5. Track improvements

## Getting Help

### Where can I find more information?

- [Documentation Index](README.md) - Overview of all guides
- [Getting Started Guide](getting-started.md) - Setup instructions
- [Troubleshooting Guide](troubleshooting.md) - Common issues
- Main [README](../README.md) - Project overview

### How do I report issues?

Check the project repository for issue reporting guidelines. Include:
- Description of the issue
- Steps to reproduce
- Error messages
- System information

### Can I contribute?

Check the project repository for contribution guidelines and code of conduct.

## Related Documentation

- [Getting Started Guide](getting-started.md) - Installation and setup
- [Data Import Guide](data-import-guide.md) - Importing data
- [Data Quality Guide](data-quality-guide.md) - Quality assessment
- [Troubleshooting Guide](troubleshooting.md) - Problem solving

---

*Have a question not answered here? Check the [Troubleshooting Guide](troubleshooting.md) or consult the specific feature guides.*

