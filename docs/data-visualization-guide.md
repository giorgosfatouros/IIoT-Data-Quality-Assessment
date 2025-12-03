# Data Visualization Guide

This guide explains how to create visualizations, explore sensor data through charts, and use advanced analytics features.

## Overview

The Data Visualization page provides comprehensive tools to:
- Create interactive charts for sensor data
- Analyze correlations between sensors
- Perform time series analysis
- Detect anomalies and trends
- Generate statistical summaries
- Export visualizations

## Accessing Data Visualization

1. Open the application at http://localhost:5173
2. Click **Data Visualization** in the sidebar or home page
3. Select a machine group from the dropdown

## Selecting Sensors for Analysis

### Step 1: Choose Machine Group

1. Select a machine group from the **Select Machine Group** dropdown
2. The system loads available sensors automatically

### Step 2: Select Sensors

1. Use the sensor selector to choose sensors for visualization
2. You can select multiple sensors for comparison
3. Selected sensors appear in the visualization

**Tips:**
- Start with 2-3 sensors for clarity
- Select related sensors (e.g., all pressure sensors)
- Use sensor categories to guide selection

### Step 3: Apply Date Range Filter (Optional)

1. Set **Date From** to filter start date
2. Set **Date To** to filter end date
3. Click **Apply Filter** to update visualizations
4. Use **Clear Filter** to remove date restrictions

**Date Range Benefits:**
- Focus on specific time periods
- Improve performance for large datasets
- Analyze seasonal patterns

## Chart Types

### Time Series Charts

**Purpose**: Show sensor values over time

**Features:**
- Multiple sensors on same chart
- Interactive tooltips showing exact values
- Zoom and pan capabilities
- Legend to toggle sensor visibility

**When to use:**
- Track trends over time
- Compare multiple sensors
- Identify patterns and cycles

### Correlation Analysis

**Purpose**: Understand relationships between sensors

**Features:**
- Correlation matrix heatmap
- Correlation coefficients (-1 to +1)
- Strong correlations highlighted

**Interpretation:**
- **+1.0**: Perfect positive correlation
- **0.0**: No correlation
- **-1.0**: Perfect negative correlation
- **>0.7 or <-0.7**: Strong correlation

**When to use:**
- Find related sensors
- Understand sensor dependencies
- Detect redundant measurements

## Advanced Analytics

### Summary Statistics

Provides statistical overview for each sensor:

- **Count**: Number of data points
- **Mean**: Average value
- **Standard Deviation**: Data spread
- **Min/Max**: Minimum and maximum values
- **Quartiles**: 25th, 50th (median), 75th percentiles

**Access:**
1. Select sensors
2. Click **Summary Stats** in Advanced Analytics
3. View statistics table

**Use cases:**
- Quick data overview
- Identify outliers
- Understand data distribution

### Time Series Analysis

Advanced time series features:

- **Rolling Mean**: Moving average over time window
- **Rolling Standard Deviation**: Volatility over time
- **Trend Detection**: Identify upward/downward trends
- **Seasonal Patterns**: Detect recurring patterns

**Access:**
1. Select sensors
2. Click **Time Series** in Advanced Analytics
3. View time series charts with trend lines

**Use cases:**
- Identify long-term trends
- Detect seasonal patterns
- Understand data volatility

### Distribution Plots

**Histogram**: Shows value distribution

- **Bins**: Number of intervals
- **Frequency**: Count of values in each bin
- **Density**: Normalized frequency

**Access:**
1. Select sensors
2. Click **Distribution** in Advanced Analytics
3. View histogram charts

**Interpretation:**
- **Normal distribution**: Bell-shaped curve
- **Skewed**: Asymmetric distribution
- **Bimodal**: Two peaks (may indicate two states)

**Use cases:**
- Understand value distribution
- Detect data quality issues
- Identify normal operating ranges

### Box Plots

**Purpose**: Show data distribution and outliers

**Components:**
- **Box**: Interquartile range (25th to 75th percentile)
- **Median Line**: Middle value
- **Whiskers**: Data range (excluding outliers)
- **Outliers**: Points beyond whiskers

**Access:**
1. Select sensors
2. Click **Box Plot** in Advanced Analytics
3. View box plot visualization

**Interpretation:**
- **Box size**: Data spread
- **Outliers**: Unusual values
- **Median position**: Data center

**Use cases:**
- Compare sensor distributions
- Identify outliers
- Understand data variability

### Seasonal Decomposition

**Purpose**: Break down time series into components

**Components:**
- **Trend**: Long-term direction
- **Seasonal**: Recurring patterns
- **Residual**: Random variation

**Access:**
1. Select sensors
2. Click **Seasonal** in Advanced Analytics
3. View decomposition charts

**Use cases:**
- Understand seasonal patterns
- Separate trend from seasonality
- Forecast future values

### Anomaly Detection

**Purpose**: Identify unusual data points

**Method**: Z-score based detection

- **Z-score**: Number of standard deviations from mean
- **Threshold**: Default 2.0 (configurable)
- **Anomalies**: Points beyond threshold

**Access:**
1. Select sensors
2. Click **Anomalies** in Advanced Analytics
3. View anomaly detection results

**Interpretation:**
- **Red points**: Detected anomalies
- **Z-score > 2**: Unusually high
- **Z-score < -2**: Unusually low

**Use cases:**
- Detect sensor malfunctions
- Identify unusual events
- Quality control

## Working with Visualizations

### Interactive Features

- **Hover**: See exact values in tooltips
- **Zoom**: Click and drag to zoom in
- **Pan**: Drag to move around zoomed view
- **Reset**: Click reset to return to full view
- **Toggle**: Click legend items to show/hide sensors

### Exporting Visualizations

**Export Options:**
- **Screenshot**: Use browser screenshot tools
- **Data Export**: Export underlying data as CSV
- **Print**: Use browser print function

**Tips:**
- Use full-screen mode for better screenshots
- Export data for external analysis tools
- Save important visualizations for reports

## Best Practices

1. **Start Simple**: Begin with time series charts
2. **Select Relevant Sensors**: Choose sensors related to your analysis
3. **Use Date Filters**: Focus on specific time periods
4. **Compare Related Sensors**: Group by category (pressure, temperature, etc.)
5. **Check Multiple Views**: Use different chart types for comprehensive understanding
6. **Interpret Statistics**: Understand what metrics mean for your use case

## Analysis Workflow

### Recommended Workflow

1. **Select Sensors**: Choose sensors of interest
2. **View Time Series**: Understand overall trends
3. **Check Summary Stats**: Get statistical overview
4. **Analyze Correlations**: Find relationships
5. **Detect Anomalies**: Identify unusual values
6. **Review Distribution**: Understand value ranges
7. **Export Results**: Save for reporting

### Example Analysis

**Scenario**: Analyzing pressure sensor data

1. Select all pressure sensors
2. View time series to see trends
3. Check correlation to find related sensors
4. Use box plots to compare distributions
5. Detect anomalies for quality issues
6. Export findings for report

## Performance Considerations

### Large Datasets

- Use date range filters to reduce data volume
- Select fewer sensors for faster rendering
- Consider using preprocessed/aggregated data

### Optimization Tips

- Limit sensor selection to essential ones
- Use appropriate date ranges
- Allow charts to fully load before interacting

## Troubleshooting

### Charts Not Displaying

**Problem**: No charts appear after selecting sensors
**Solutions:**
- Check data is loaded (see [Data Loading Guide](data-loading-guide.md))
- Verify sensors have data in selected date range
- Check browser console for errors
- Refresh the page

### Slow Performance

**Problem**: Charts load slowly
**Solutions:**
- Reduce date range
- Select fewer sensors
- Use aggregated data source
- Check network connection

### Missing Analytics

**Problem**: Advanced analytics not available
**Solutions:**
- Ensure sensors are selected
- Check data has sufficient points
- Verify backend service is running
- Review error messages

## Next Steps

After creating visualizations:

1. **Assess Quality**: Use [Data Quality Guide](data-quality-guide.md) for comprehensive assessment
2. **Find Missing Values**: Check [Missing Values Guide](missing-values-guide.md)
3. **Detect Invalid Values**: See [Invalid Values Guide](invalid-values-guide.md)
4. **Chat with Agent**: Use [DQA Agent Guide](dqa-agent-guide.md) for insights

## Related Documentation

- [Data Loading Guide](data-loading-guide.md) - Select and load data
- [Data Quality Guide](data-quality-guide.md) - Comprehensive quality assessment
- [Missing Values Guide](missing-values-guide.md) - Missing data analysis

---

*For API details, see the [Backend API Documentation](../backend/README.md).*

