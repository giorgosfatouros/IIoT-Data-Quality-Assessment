# Data Loading Guide

This guide explains how to select machine groups, view sensor data, and understand metadata in the Data Loading page.

## Overview

The Data Loading page is your starting point for working with sensor data. It allows you to:
- Select machine groups (tables) that contain your sensor data
- View raw sensor data from the database
- View preprocessed/aggregated sensor data
- Understand sensor metadata and thresholds
- Check aggregation frequency settings

## Accessing Data Loading

1. Open the application at http://localhost:5173
2. Click **Data Loading** in the sidebar or home page
3. You'll see two tabs:
   - **Load from Database** - Select and view existing data
   - **Upload Data** - Import new data (see [Data Import Guide](data-import-guide.md))

## Selecting a Machine Group

### Step 1: Choose a Table

1. In the **Load from Database** tab, find the **Select Machine Group** dropdown
2. Click the dropdown to see available machine groups
3. Select the machine group you want to analyze

**Available tables:**
- Tables are created when you import data (see [Data Import Guide](data-import-guide.md))
- Table names correspond to machine group identifiers
- Example: `KT2201`, `K3301`, `K5700`

### Step 2: View Data

After selecting a table, the system automatically loads:
- **Aggregation Frequency**: How often data is aggregated
- **Available Sensors**: List of sensors in the selected machine group
- **Raw Data**: Original sensor readings
- **Preprocessed Data**: Aggregated/processed sensor data

## Understanding Data Views

### Raw Sensor Data

Raw data shows the original sensor readings as imported:

- **Timestamp**: Time of each reading
- **Sensor Columns**: One column per sensor tag
- **Values**: Actual sensor readings at each timestamp
- **Missing Values**: Shown as empty cells or `NaN`

**When to use:**
- Viewing original data before processing
- Checking data completeness
- Verifying import accuracy

**Viewing Raw Data:**
1. Select a machine group
2. Click **Show Raw Data** toggle
3. Data appears in a table format
4. Use search to filter by sensor or timestamp

### Preprocessed Data

Preprocessed data shows aggregated sensor readings:

- **Aggregated Values**: Data processed by aggregation rules
- **Reduced Frequency**: Data sampled at aggregation intervals
- **Applied Rules**: Min, max, avg, or sum based on sensor configuration

**When to use:**
- Analyzing trends over time
- Reducing data volume for visualization
- Working with aggregated insights

**Viewing Preprocessed Data:**
1. Select a machine group
2. Preprocessed data loads automatically
3. Click **Show Preprocessed Data** to view
4. Data appears in table format

## Aggregation Frequency

The aggregation frequency indicates how often sensor data is aggregated:

- **Display**: Shown in seconds (e.g., `3600` = 1 hour)
- **Purpose**: Determines time intervals for aggregated data
- **Configuration**: Set during worker service configuration

**Understanding the value:**
- `3600` seconds = 1 hour aggregation window
- `600` seconds = 10 minute aggregation window
- Lower values = more frequent aggregation = more data points

**Note**: Aggregation frequency affects the granularity of preprocessed data and visualizations.

## Sensor Metadata

Sensor metadata provides information about each sensor:

### Viewing Metadata

1. Select a machine group
2. Click **Show Tags** toggle
3. View metadata table with sensor information

### Metadata Fields

Each sensor has the following metadata:

- **TAG**: Unique sensor identifier (e.g., `22PI102`)
- **Tag Description**: Human-readable description
- **Machine Group**: Machine group identifier
- **Low Threshold**: Lower limit for valid readings
- **High Threshold**: Upper limit for valid readings
- **Threshold Type**: `Up`, `Down`, or `Up/Down`
- **Aggregation Rule**: How data is aggregated (`min`, `max`, `avg`, `sum`)
- **Engineering Units**: Measurement units (e.g., `Kgf/cm2`, `degC`)
- **Category**: Sensor category (e.g., `Pressure`, `Temperature`)

### Using Metadata

Metadata helps you:
- Understand what each sensor measures
- Know valid value ranges (thresholds)
- Understand how data is processed
- Identify sensor categories

## Data Preview Features

### Search and Filter

**Search Raw Data:**
- Use the search box to filter by sensor tag or timestamp
- Searches across all columns
- Real-time filtering as you type

**Search Preprocessed Data:**
- Similar search functionality
- Filters aggregated data table

### Data Export

Export data for external analysis:

1. Select your data view (raw or preprocessed)
2. Click **Export Data** button (if available)
3. Data downloads as CSV file

**Note**: Export functionality may vary based on data size and browser capabilities.

## Data Source Selection

The Data Loading page allows you to choose data sources:

- **Database**: Load from TimescaleDB (default)
- **Upload**: Import new data files (see [Data Import Guide](data-import-guide.md))

## Understanding Data Structure

### Timestamp Column

- **Format**: `YYYY-MM-DD HH:MM:SS` or ISO format
- **Purpose**: Time reference for all sensor readings
- **Required**: First column in raw data

### Sensor Columns

- **Naming**: Matches sensor TAG from metadata
- **Values**: Numeric sensor readings
- **Missing**: Empty or `NaN` for missing values
- **Units**: Defined in metadata (Engineering Units)

### Data Rows

- Each row represents one time point
- Contains timestamp and all sensor readings
- Rows are ordered chronologically

## Tips and Best Practices

1. **Start with Preprocessed Data**: Easier to work with for initial analysis
2. **Check Aggregation Frequency**: Understand data granularity
3. **Review Metadata**: Know sensor thresholds and units
4. **Use Search**: Quickly find specific sensors or time ranges
5. **Verify Data Completeness**: Check for missing values in raw data

## Common Tasks

### Viewing Specific Sensors

1. Select machine group
2. Use search to filter by sensor tag
3. Or navigate to Data Visualization to select specific sensors

### Checking Data Range

1. View raw or preprocessed data
2. Check first and last timestamps
3. Verify date range matches expectations

### Understanding Thresholds

1. Click **Show Tags** to view metadata
2. Find sensor of interest
3. Check Low/High Threshold values
4. Note Threshold Type (Up/Down/Up-Down)

## Next Steps

After loading your data:

1. **Visualize**: Go to [Data Visualization Guide](data-visualization-guide.md) to create charts
2. **Check Quality**: Use [Data Quality Guide](data-quality-guide.md) for quality assessment
3. **Find Missing Values**: See [Missing Values Guide](missing-values-guide.md)
4. **Detect Invalid Values**: Check [Invalid Values Guide](invalid-values-guide.md)

## Troubleshooting

### No Tables Available

**Problem**: Dropdown is empty
**Solutions**:
- Import data first (see [Data Import Guide](data-import-guide.md))
- Check database connection
- Verify backend service is running

### Data Not Loading

**Problem**: Selected table but no data appears
**Solutions**:
- Check backend logs: `docker-compose logs backend`
- Verify table exists in database
- Check for error messages in UI
- Refresh the page

### Missing Sensors

**Problem**: Expected sensors not visible
**Solutions**:
- Verify sensors were imported (check import logs)
- Check sensor tags match metadata
- Review data import process

## Related Documentation

- [Data Import Guide](data-import-guide.md) - Import new sensor data
- [Data Visualization Guide](data-visualization-guide.md) - Create visualizations
- [Data Quality Guide](data-quality-guide.md) - Assess data quality

---

*For technical details, see the [Backend API Documentation](../backend/README.md).*

