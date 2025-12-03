# Data Import Guide

This guide explains how to import sensor data into the IIoT Data Quality Assessment Service using CSV files.

## Overview

The data import feature allows you to upload CSV files containing:
- **Sensor Data**: Time-series readings from your IIoT sensors
- **Tags Metadata**: Sensor definitions, thresholds, and metadata

The system validates your files, detects the machine type, and imports the data into the database for analysis.

## File Format Requirements

### Sensor Data File (data.csv)

The sensor data CSV file should contain:
- A `timestamp` column with date/time values
- One column for each sensor tag (sensor identifier)
- Each row represents a time point with sensor readings

**Example structure:**
```csv
timestamp,22PI102,22PI103,22TI111,22TI113
2024-01-01 00:00:00,8.5,7.2,85.3,82.1
2024-01-01 00:00:10,8.6,7.3,85.5,82.3
2024-01-01 00:00:20,8.4,7.1,85.2,82.0
```

**Requirements:**
- First column must be `timestamp` (case-sensitive)
- Timestamp format: `YYYY-MM-DD HH:MM:SS` or ISO format
- Column names should match sensor tags in the tags file
- Missing values can be empty cells or `NaN`
- File must be valid CSV format

### Tags Metadata File (tags.csv)

The tags file describes each sensor with metadata including thresholds and aggregation rules.

**Required columns:**
- `TAG` - Unique sensor identifier (must match data file column names)
- `Tag Description` - Human-readable description
- `MACHINE_GROUP` - Machine group identifier
- `LOW_THRESHOLD` - Lower threshold value (can be empty)
- `HIGH_THRESHOLD` - Upper threshold value (can be empty)
- `THRESHOLD_TYPE` - One of: `Up`, `Down`, or `Up/Down`
- `AGGREGATION_RULE` - Aggregation method: `min`, `max`, `avg`, `sum`
- `ENGINEERING_UNITS` - Measurement units (e.g., `Kgf/cm2`, `degC`)
- `CATEGORY` - Sensor category (e.g., `Pressure`, `Temperature`)

**Example tags.csv:**
```csv
TAG,Tag Description,MACHINE_GROUP,LOW_THRESHOLD,HIGH_THRESHOLD,THRESHOLD_TYPE,AGGREGATION_RULE,ENGINEERING_UNITS,CATEGORY
22PI102,SEAL OIL MAIN PUMP PRESSURE,K-2201_KT-2201,6,,Down,min,Kgf/cm2,Pressure
22PI103,CONTROL OIL HEADER PRESSURE,K-2201_KT-2201,5,,Down,min,Kgf/cm2,Pressure
22TI111,ST. TURBINE N.D.E BEARING TEMPERATURE,K-2201_KT-2201,,120,Up,max,degC,Temperature
```

**Threshold Types:**
- `Down` - Alarm when value falls below LOW_THRESHOLD
- `Up` - Alarm when value exceeds HIGH_THRESHOLD
- `Up/Down` - Alarm when value is outside the threshold range

## Import Process

### Step 1: Navigate to Data Loading

1. Open the application at http://localhost:5173
2. Click on **Data Loading** in the sidebar or home page
3. Select the **Upload Data** tab

### Step 2: Prepare Your Files

Ensure your CSV files are ready:
- Data file contains sensor readings with timestamps
- Tags file contains all required columns
- Sensor tags in data file match TAG values in tags file

### Step 3: Validate Files (Optional but Recommended)

Before importing, validate your files:

1. Click **Choose Data File** and select your sensor data CSV
2. Click **Choose Tags File** and select your tags CSV
3. Click **Validate Files** button
4. Review validation results:
   - File format checks
   - Required columns presence
   - Data type validation
   - Machine type detection

**If validation fails:**
- Review the error messages
- Fix issues in your CSV files
- Re-validate until all checks pass

### Step 4: Configure Import Settings

1. **Table Name**: Enter a name for your machine group (e.g., `KT2201`)
   - Will be converted to uppercase automatically
   - Must be unique if not overwriting

2. **Machine Type**: Select from dropdown or use `AUTO`
   - `AUTO` - System detects from filename or data patterns
   - `KT2201` - K-2201/KT-2201 Machine
   - `K3301` - K-3301/KT-3301 Machine
   - `K5700` - K-5700 Machine

3. **Overwrite Existing**: 
   - Check to replace existing table with same name
   - Leave unchecked to prevent accidental overwrites

4. **Select Sensors** (Optional):
   - Leave empty to import all sensors
   - Enter comma-separated sensor tags to import only specific sensors
   - Example: `22PI102,22PI103,22TI111`

### Step 5: Start Import

1. Click **Upload and Import** button
2. The system will:
   - Validate files again
   - Create an import job
   - Start background import process
   - Return a job ID for tracking

### Step 6: Monitor Import Progress

After starting the import, you'll see:
- **Job ID**: Unique identifier for tracking
- **Status**: Current import status (Pending, Running, Completed, Failed)
- **Progress**: Percentage complete
- **Rows Processed**: Number of rows imported

**Status updates:**
- `PENDING` - Job created, waiting to start
- `RUNNING` - Import in progress
- `COMPLETED` - Import finished successfully
- `FAILED` - Import failed (check error message)

**To check status:**
- The UI automatically refreshes status
- Or manually refresh the status display
- Check backend logs if issues occur: `docker-compose logs backend`

## Import Status Tracking

The import runs asynchronously in the background. You can:

1. **Monitor in UI**: Status updates automatically in the Data Loading page
2. **Check Job Status**: Use the job ID to query status via API
3. **View Logs**: Check backend service logs for detailed progress

**Typical import times:**
- Small datasets (< 10,000 rows): 1-2 minutes
- Medium datasets (10,000 - 100,000 rows): 5-10 minutes
- Large datasets (> 100,000 rows): 10+ minutes (depends on data size)

## Supported Machine Types

The system supports automatic detection for:

- **KT2201** / **K-2201** / **KT-2201**: Detected from filename patterns or column names
- **K3301** / **K-3301** / **KT-3301**: Detected from filename patterns
- **K5700**: Detected from filename patterns
- **AUTO**: System attempts to detect from file patterns

If auto-detection fails, manually select the machine type.

## Import Options

### Selective Sensor Import

You can import only specific sensors by providing a comma-separated list:

```
22PI102,22PI103,22TI111
```

This is useful when:
- You only need certain sensors for analysis
- Testing with a subset of data
- Reducing import time for large datasets

### Overwrite Existing Data

When checked, the import will:
- Drop existing table if it exists
- Create a new table with imported data
- **Warning**: This permanently deletes existing data

Use with caution! Consider backing up data before overwriting.

## Troubleshooting Import Issues

### Validation Errors

**Problem**: Files fail validation
**Solutions**:
- Check CSV format (no extra commas, proper quoting)
- Verify required columns are present
- Ensure timestamp column is named exactly `timestamp`
- Check data types match expected formats

### Import Fails

**Problem**: Import job fails
**Solutions**:
- Check backend logs: `docker-compose logs backend`
- Verify database connection is available
- Ensure sufficient disk space
- Check file encoding (should be UTF-8)

### Machine Type Not Detected

**Problem**: Auto-detection fails
**Solutions**:
- Manually select machine type from dropdown
- Check filename contains machine identifier
- Verify data columns match expected patterns

### Slow Import

**Problem**: Import takes too long
**Solutions**:
- Large files take time - be patient
- Check database performance
- Consider importing subset of sensors first
- Verify worker service is running

## Best Practices

1. **Always validate first**: Use validation before importing to catch errors early
2. **Start small**: Test with a small dataset first
3. **Backup data**: Export existing data before overwriting
4. **Check file encoding**: Ensure CSV files are UTF-8 encoded
5. **Verify timestamps**: Ensure timestamp format is consistent
6. **Match sensor tags**: Ensure data file columns match tags file TAG values
7. **Monitor progress**: Keep an eye on import status for large files

## Next Steps

After successful import:

1. **Load Data**: Go to [Data Loading Guide](data-loading-guide.md) to view your imported data
2. **Visualize**: Use [Data Visualization Guide](data-visualization-guide.md) to explore your data
3. **Assess Quality**: Check [Data Quality Guide](data-quality-guide.md) for quality metrics

## Related Documentation

- [Data Loading Guide](data-loading-guide.md) - View and explore imported data
- [Getting Started Guide](getting-started.md) - Initial setup
- [Troubleshooting Guide](troubleshooting.md) - Common issues and solutions

---

*For API-based import, see the [Backend API Documentation](../backend/README.md).*

