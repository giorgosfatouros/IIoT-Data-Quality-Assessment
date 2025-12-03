# Invalid Values Guide

This guide explains how to detect, analyze, and understand invalid sensor readings based on threshold violations.

## Overview

Invalid values analysis helps you:
- Identify sensor readings outside acceptable ranges
- Detect threshold violations (alarms)
- Understand sensor behavior patterns
- Assess data accuracy
- Identify sensors needing calibration or maintenance

## Understanding Invalid Values

### What are Invalid Values?

Invalid values are sensor readings that:
- Fall below low threshold (for "Down" threshold type)
- Exceed high threshold (for "Up" threshold type)
- Fall outside threshold range (for "Up/Down" threshold type)
- Indicate sensor malfunction or abnormal conditions

### Threshold Types

**Down Threshold:**
- Alarm when value **falls below** LOW_THRESHOLD
- Example: Pressure sensor alarm when pressure < 6 Kgf/cm2

**Up Threshold:**
- Alarm when value **exceeds** HIGH_THRESHOLD
- Example: Temperature sensor alarm when temperature > 120 degC

**Up/Down Threshold:**
- Alarm when value is **outside** the range
- Example: Alarm when value < LOW_THRESHOLD or > HIGH_THRESHOLD

### Why Analyze Invalid Values?

- **Data Accuracy**: Identify inaccurate readings
- **Sensor Health**: Detect malfunctioning sensors
- **Safety**: Identify critical threshold violations
- **Maintenance**: Plan sensor calibration or replacement
- **Quality Control**: Ensure data meets quality standards

## Accessing Invalid Values Analysis

1. Open the application at http://localhost:5173
2. Click **Invalid Values** in the sidebar or home page
3. Select a machine group from the dropdown

## Performing Invalid Values Analysis

### Step 1: Select Machine Group

1. Choose a machine group from the **Select Machine Group** dropdown
2. The system loads available sensors automatically

### Step 2: Select Sensors

1. Use the sensor selector to choose sensors for analysis
2. You can select multiple sensors
3. Selected sensors must have threshold definitions

**Note**: Only sensors with defined thresholds can be analyzed.

### Step 3: Configure Threshold

1. Set **Alarm Threshold Percentage** (default: 16.67%)
2. This determines which sensors are flagged
3. Sensors exceeding this alarm rate are highlighted

**Threshold Percentage:**
- Percentage of readings that are alarms
- Example: 16.67% means 1 in 6 readings is an alarm
- Adjust based on your quality requirements

### Step 4: Run Analysis

1. Click **Analyze Invalid Values** button
2. Wait for analysis to complete
3. Results appear in multiple sections

## Understanding Analysis Results

### Overall Statistics

**Summary Information:**
- **Table Name**: Machine group analyzed
- **Selected Sensors**: Sensors included in analysis
- **Total Readings**: Total number of sensor readings
- **Total Alarms**: Count of threshold violations
- **Average Alarms per Sensor**: Mean alarm count
- **Max Alarms Sensor**: Sensor with most alarms
- **Max Alarms Count**: Highest alarm count

**Interpretation:**
- **Low alarm rate** (< 5%): Good data quality
- **Moderate alarm rate** (5-15%): Some concerns, investigate
- **High alarm rate** (> 15%): Significant issues, take action

### Sensor Statistics

**Per-Sensor Information:**
- **Sensor Name**: Sensor identifier
- **Total Readings**: Readings for this sensor
- **Total Alarms**: Count of threshold violations
- **Alarm Percentage**: Percentage of invalid readings
- **Metadata**: Sensor description, thresholds, units

**Visualization:**
- Bar chart showing alarm percentage per sensor
- Sorted by alarm count (highest first)
- Color-coded for quick identification

**Use Cases:**
- Identify sensors with most alarms
- Compare alarm rates across sensors
- Prioritize sensors for investigation

### Time Series Analysis

**Time Series Charts:**
- Show sensor values over time
- Highlight invalid readings (alarms)
- Display threshold lines
- Show alarm count over time

**Features:**
- **Original Values**: Actual sensor readings
- **Rolling Mean**: Moving average
- **Rolling Std**: Volatility indicator
- **Alarm Count**: Number of alarms over time
- **Threshold Lines**: Visual reference for valid range

**Interpretation:**
- **Spikes**: Sudden threshold violations
- **Trends**: Gradual drift toward thresholds
- **Patterns**: Recurring alarm times
- **Clusters**: Groups of alarms indicating issues

### Invalid Reading Points

**Detailed Alarm Information:**
- **Timestamp**: When alarm occurred
- **Value**: Sensor reading that triggered alarm
- **Alarm Count**: Number of alarms at this time point

**Use Cases:**
- Identify specific alarm events
- Understand alarm timing
- Correlate alarms with other events
- Document alarm occurrences

## Interpreting Results

### Good Data Quality Indicators

- **Low alarm rate** (< 5% overall)
- **Even distribution** across sensors
- **Random alarms** (not systematic)
- **Alarms within expected range**

### Data Quality Concerns

- **High alarm rate** (> 15% overall)
- **Concentrated in specific sensors**
- **Systematic patterns** (same time, same conditions)
- **Trending toward thresholds**

### Action Items Based on Results

**High Alarm Rate:**
1. Check sensor calibration
2. Verify threshold settings
3. Investigate sensor hardware
4. Review operating conditions

**Systematic Patterns:**
1. Identify root cause
2. Check environmental factors
3. Review maintenance schedules
4. Adjust thresholds if appropriate

**Specific Sensor Issues:**
1. Inspect sensor hardware
2. Check sensor connections
3. Verify sensor placement
4. Consider sensor replacement

## Working with Thresholds

### Understanding Threshold Settings

Thresholds are defined in the tags metadata file:
- **LOW_THRESHOLD**: Lower limit
- **HIGH_THRESHOLD**: Upper limit
- **THRESHOLD_TYPE**: Up, Down, or Up/Down

### Adjusting Analysis Threshold

The alarm threshold percentage in the UI:
- Controls which sensors are flagged
- Does not change sensor thresholds
- Helps focus on most problematic sensors

### Best Practices

1. **Review Thresholds**: Ensure thresholds are appropriate
2. **Document Changes**: Record threshold modifications
3. **Regular Updates**: Adjust thresholds as needed
4. **Validate Settings**: Verify threshold accuracy

## Common Scenarios

### Scenario 1: Single Sensor with High Alarm Rate

**Observation**: One sensor has 25% alarm rate
**Action**:
- Check sensor calibration
- Verify threshold settings
- Inspect sensor hardware
- Review sensor placement

### Scenario 2: All Sensors Alarming at Same Time

**Observation**: Multiple sensors alarm simultaneously
**Action**:
- Check system-wide conditions
- Review environmental factors
- Investigate system events
- Verify threshold settings

### Scenario 3: Gradual Drift Toward Threshold

**Observation**: Values trending toward threshold over time
**Action**:
- Monitor trend closely
- Plan preventive maintenance
- Check for gradual degradation
- Consider threshold adjustment

### Scenario 4: Periodic Alarm Patterns

**Observation**: Alarms occur at regular intervals
**Action**:
- Identify pattern cause
- Check scheduled operations
- Review system cycles
- Investigate environmental cycles

## Best Practices

1. **Regular Analysis**: Check invalid values periodically
2. **Set Standards**: Define acceptable alarm rates
3. **Document Findings**: Record alarm statistics
4. **Investigate Causes**: Determine why alarms occur
5. **Take Action**: Address sensors with high alarm rates
6. **Monitor Trends**: Track alarm rates over time

## Exporting Results

**Export Options:**
- **Screenshot**: Capture analysis results
- **Data Export**: Export alarm data as CSV (if available)
- **Report**: Document findings for stakeholders

## Troubleshooting

### No Invalid Values Detected

**Problem**: Analysis shows 0% alarms
**Possible Causes**:
- Data is truly valid (good!)
- Thresholds not defined for sensors
- Thresholds set too wide
- Check threshold definitions

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
- Verify threshold definitions
- Check sensor selection
- Review date range
- Compare with raw data

## Next Steps

After analyzing invalid values:

1. **Check Missing Values**: See [Missing Values Guide](missing-values-guide.md)
2. **Assess Overall Quality**: Use [Data Quality Guide](data-quality-guide.md)
3. **Visualize Data**: Go to [Data Visualization Guide](data-visualization-guide.md)
4. **Chat with Agent**: Ask [DQA Agent](dqa-agent-guide.md) about invalid values

## Related Documentation

- [Data Quality Guide](data-quality-guide.md) - Comprehensive quality assessment
- [Missing Values Guide](missing-values-guide.md) - Missing data analysis
- [Data Loading Guide](data-loading-guide.md) - Load and view data

---

*For technical details, see the [Backend API Documentation](../backend/README.md).*

