# Missing Values Guide

This guide explains how to identify, analyze, and understand missing values in your sensor data.

## Overview

Missing values analysis helps you:
- Identify sensors with incomplete data
- Quantify missing data percentages
- Find missing data patterns over time
- Understand data completeness
- Make informed decisions about data quality

## Understanding Missing Values

### What are Missing Values?

Missing values occur when:
- Sensor readings are not recorded at expected timestamps
- Data collection was interrupted
- Sensors malfunctioned temporarily
- Data transmission failed

### Why Analyze Missing Values?

- **Data Quality Assessment**: Understand data completeness
- **Sensor Health**: Identify malfunctioning sensors
- **Analysis Reliability**: Know which sensors have sufficient data
- **Maintenance Planning**: Identify sensors needing attention

## Accessing Missing Values Analysis

1. Open the application at http://localhost:5173
2. Click **Missing Values** in the sidebar or home page
3. Select a machine group from the dropdown

## Performing Missing Values Analysis

### Step 1: Select Machine Group

1. Choose a machine group from the **Select Machine Group** dropdown
2. The system loads available sensors automatically

### Step 2: Select Sensors (Optional)

1. Use the sensor selector to choose specific sensors
2. Leave empty to analyze all sensors
3. Selected sensors appear in the analysis

**Tip**: Start with all sensors for overview, then focus on specific sensors if needed.

### Step 3: Run Analysis

1. Click **Analyze Missing Values** button
2. Wait for analysis to complete
3. Results appear in multiple sections

## Understanding Analysis Results

### Overall Statistics

**General Information:**
- **Date Range**: Start and end dates of data
- **Total Data Points**: Expected number of readings
- **Total Missing Values**: Count of missing readings
- **Missing Percentage**: Overall completeness percentage
- **Number of Sensors**: Sensors analyzed

**Interpretation:**
- **< 5% missing**: Excellent data completeness
- **5-10% missing**: Good, minor gaps
- **10-20% missing**: Moderate, some concerns
- **> 20% missing**: Poor, significant data gaps

### Missing Values by Sensor

**Per-Sensor Statistics:**
- **Sensor Name**: Sensor identifier
- **Data Points**: Expected readings
- **Actual Readings**: Recorded readings
- **Missing Readings**: Count of missing values
- **Missing Percentage**: Sensor-specific completeness

**Visualization:**
- Bar chart showing missing percentage per sensor
- Sorted by missing percentage (highest first)
- Color-coded for quick identification

**Use Cases:**
- Identify sensors with most missing data
- Compare completeness across sensors
- Prioritize sensors for investigation

### Missing Intervals

**Interval Information:**
- **Start Time**: When missing period began
- **End Time**: When missing period ended
- **Duration**: Length of missing period (hours/days)
- **Sensor**: Which sensor had the gap

**Longest Intervals:**
- Shows sensors with longest continuous missing periods
- Helps identify major data collection issues
- Useful for understanding system downtime

**Interpretation:**
- **Short intervals (< 1 hour)**: Minor gaps, likely normal
- **Medium intervals (1-24 hours)**: Moderate issues
- **Long intervals (> 24 hours)**: Major problems, investigate

### Missing Values Over Time

**Time Series Visualization:**
- Chart showing missing values across time
- Identifies patterns in missing data
- Shows when data collection issues occurred

**Patterns to Look For:**
- **Random gaps**: Normal sensor behavior
- **Periodic gaps**: Scheduled maintenance or system issues
- **Continuous gaps**: Sensor malfunction or system downtime
- **Increasing gaps**: Degrading sensor or system

## Interpreting Results

### Good Data Quality Indicators

- **Low missing percentage** (< 5% overall)
- **Even distribution** across sensors
- **Short, infrequent gaps**
- **No systematic patterns**

### Data Quality Concerns

- **High missing percentage** (> 20% overall)
- **Concentrated in specific sensors**
- **Long continuous gaps**
- **Systematic patterns** (same time daily, etc.)

### Action Items Based on Results

**High Missing Percentage:**
1. Investigate sensor hardware
2. Check data collection system
3. Review transmission logs
4. Consider sensor replacement

**Long Missing Intervals:**
1. Identify root cause (maintenance, failure, etc.)
2. Document system downtime
3. Plan preventive measures
4. Consider backup sensors

**Systematic Patterns:**
1. Review maintenance schedules
2. Check system configuration
3. Investigate environmental factors
4. Adjust data collection settings

## Best Practices

1. **Regular Analysis**: Check missing values periodically
2. **Set Thresholds**: Define acceptable missing percentages
3. **Document Issues**: Record reasons for missing data
4. **Monitor Trends**: Track missing values over time
5. **Take Action**: Address sensors with high missing rates

## Working with Missing Data

### Before Analysis

- **Understand Expected Frequency**: Know how often readings should occur
- **Check Data Range**: Ensure sufficient data for analysis
- **Review Sensor Status**: Know which sensors are active

### During Analysis

- **Focus on High Missing Rates**: Prioritize sensors with most issues
- **Check Time Patterns**: Look for systematic gaps
- **Compare Sensors**: Identify outliers

### After Analysis

- **Document Findings**: Record missing value statistics
- **Investigate Causes**: Determine why data is missing
- **Plan Improvements**: Address identified issues
- **Re-analyze**: Check improvements after fixes

## Common Scenarios

### Scenario 1: One Sensor with High Missing Rate

**Observation**: Single sensor has 30% missing values
**Action**: 
- Check sensor hardware
- Review sensor-specific logs
- Consider sensor replacement

### Scenario 2: All Sensors Missing at Same Time

**Observation**: All sensors show gaps at same timestamps
**Action**:
- Check data collection system
- Review system maintenance logs
- Investigate network issues

### Scenario 3: Periodic Missing Patterns

**Observation**: Missing values occur at same time daily
**Action**:
- Check scheduled maintenance
- Review system configuration
- Investigate environmental factors

## Exporting Results

**Export Options:**
- **Screenshot**: Capture analysis results
- **Data Export**: Export statistics as CSV (if available)
- **Report**: Document findings for stakeholders

## Troubleshooting

### No Missing Values Detected

**Problem**: Analysis shows 0% missing
**Possible Causes**:
- Data is truly complete (good!)
- Analysis method doesn't detect gaps
- Data preprocessing filled gaps
- Check raw data for verification

### Analysis Takes Too Long

**Problem**: Analysis is slow
**Solutions**:
- Select fewer sensors
- Use date range filters
- Check data volume
- Verify backend performance

### Unexpected Results

**Problem**: Results don't match expectations
**Solutions**:
- Verify data source
- Check date range
- Review sensor selection
- Compare with raw data

## Next Steps

After analyzing missing values:

1. **Check Invalid Values**: See [Invalid Values Guide](invalid-values-guide.md)
2. **Assess Overall Quality**: Use [Data Quality Guide](data-quality-guide.md)
3. **Visualize Data**: Go to [Data Visualization Guide](data-visualization-guide.md)
4. **Chat with Agent**: Ask [DQA Agent](dqa-agent-guide.md) about missing values

## Related Documentation

- [Data Quality Guide](data-quality-guide.md) - Comprehensive quality assessment
- [Invalid Values Guide](invalid-values-guide.md) - Invalid readings analysis
- [Data Loading Guide](data-loading-guide.md) - Load and view data

---

*For technical details, see the [Backend API Documentation](../backend/README.md).*

