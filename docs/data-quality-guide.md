# Data Quality Guide

This guide explains how to perform comprehensive data quality assessment, understand quality metrics, and interpret quality reports.

## Overview

The Data Quality page provides a comprehensive assessment of your sensor data quality, including:
- Completeness analysis
- Accuracy assessment
- Consistency checks
- Outlier detection
- Correlation analysis
- Overall quality scores

## Understanding Data Quality

### What is Data Quality?

Data quality refers to the fitness of data for its intended use, measured by:
- **Completeness**: How much data is present
- **Accuracy**: How correct the data is
- **Consistency**: How uniform the data is
- **Timeliness**: How current the data is
- **Validity**: How well data conforms to rules

### Why Assess Data Quality?

- **Reliability**: Ensure analysis results are trustworthy
- **Decision Making**: Make informed decisions based on quality data
- **Compliance**: Meet data quality standards
- **Maintenance**: Identify sensors needing attention
- **Optimization**: Improve data collection processes

## Accessing Data Quality Assessment

1. Open the application at http://localhost:5173
2. Click **Data Quality** in the sidebar or home page
3. Select a machine group from the dropdown

## Performing Quality Assessment

### Step 1: Select Machine Group

1. Choose a machine group from the **Select Machine Group** dropdown
2. The system loads available sensors automatically

### Step 2: Apply Date Range Filter (Optional)

1. Set **Date From** to filter start date
2. Set **Date To** to filter end date
3. Click **Apply Filter** to update assessment
4. Use **Clear Filter** to remove date restrictions

**Benefits of Date Filtering:**
- Focus on specific time periods
- Compare quality across time periods
- Improve performance for large datasets

### Step 3: Run Quality Assessment

1. Click **Assess Data Quality** button
2. Wait for analysis to complete
3. Results appear in multiple sections

## Understanding Quality Metrics

### General Information

**Data Overview:**
- **Date Range**: Start and end dates of analyzed data
- **Total Data Points**: Expected number of readings
- **Total Missing Values**: Count of missing readings
- **Missing Percentage**: Overall completeness
- **Number of Sensors**: Sensors analyzed

**Interpretation:**
- **Low missing percentage** (< 5%): Excellent completeness
- **Moderate missing** (5-10%): Good, minor gaps
- **High missing** (> 10%): Concerns, investigate

### Descriptive Statistics

**Per-Sensor Statistics:**
- **Count**: Number of data points
- **Mean**: Average value
- **Standard Deviation**: Data spread
- **Min/Max**: Minimum and maximum values
- **Quartiles**: 25th, 50th (median), 75th percentiles

**Use Cases:**
- Understand data distribution
- Identify outliers
- Compare sensors
- Detect anomalies

### Completeness Check

**Completeness Metrics:**
- **Overall Completeness**: Percentage of complete data
- **Completeness Threshold**: Minimum acceptable level
- **Incomplete Sensors**: Sensors below threshold

**Interpretation:**
- **> 95%**: Excellent completeness
- **90-95%**: Good completeness
- **85-90%**: Acceptable, monitor
- **< 85%**: Poor, take action

**Action Items:**
- Investigate incomplete sensors
- Check data collection system
- Review sensor health
- Plan improvements

### Consistency Check

**Consistency Metrics:**
- **Has Duplicates**: Whether duplicate records exist
- **Duplicate Count**: Number of duplicate records
- **Duplicate Percentage**: Percentage of duplicates
- **Timestamps Consistent**: Whether timestamps are valid

**Interpretation:**
- **No duplicates**: Good consistency
- **Few duplicates** (< 1%): Acceptable
- **Many duplicates** (> 1%): Data quality issue
- **Inconsistent timestamps**: System issue

**Action Items:**
- Remove duplicates if needed
- Fix timestamp issues
- Review data collection process
- Improve data validation

### Outlier Detection

**Outlier Information:**
- **Sensor Name**: Which sensor has outliers
- **Outlier Percentage**: Percentage of outlier values

**Outlier Detection Method:**
- Uses statistical methods (Z-score, IQR)
- Identifies values significantly different from normal
- Helps detect sensor malfunctions

**Interpretation:**
- **Low outlier rate** (< 2%): Normal variation
- **Moderate outliers** (2-5%): Some concerns
- **High outliers** (> 5%): Significant issues

**Action Items:**
- Investigate high-outlier sensors
- Check sensor calibration
- Review operating conditions
- Consider sensor replacement

### Accuracy Issues

**Accuracy Metrics:**
- **Sensor Name**: Sensor with accuracy issues
- **Issues Percentage**: Percentage of inaccurate readings
- **Threshold Type**: Type of threshold violation
- **Low/High Threshold**: Threshold values

**Accuracy Assessment:**
- Based on threshold violations
- Identifies readings outside valid ranges
- Helps assess sensor accuracy

**Interpretation:**
- **Low issue rate** (< 5%): Good accuracy
- **Moderate issues** (5-15%): Some concerns
- **High issues** (> 15%): Poor accuracy

**Action Items:**
- Check sensor calibration
- Verify threshold settings
- Review sensor placement
- Consider recalibration

### Correlation Analysis

**Correlation Information:**
- **Strong Correlations**: Pairs of highly correlated sensors
- **Correlation Coefficient**: Strength of relationship (-1 to +1)
- **Correlation Matrix**: Full correlation matrix

**Correlation Interpretation:**
- **+1.0**: Perfect positive correlation
- **+0.7 to +1.0**: Strong positive correlation
- **-0.7 to -1.0**: Strong negative correlation
- **-0.3 to +0.3**: Weak correlation

**Use Cases:**
- Find related sensors
- Understand sensor dependencies
- Detect redundant measurements
- Identify sensor groups

## Quality Score Interpretation

### Overall Quality Assessment

The system provides an overall quality assessment based on:
- Completeness metrics
- Accuracy metrics
- Consistency metrics
- Outlier rates

**Quality Levels:**
- **Excellent**: All metrics within acceptable ranges
- **Good**: Minor issues, generally acceptable
- **Fair**: Some concerns, monitor closely
- **Poor**: Significant issues, take action

### Quality Report Components

**Summary Section:**
- Overall quality score
- Key findings
- Recommendations

**Detailed Metrics:**
- Per-sensor statistics
- Completeness details
- Accuracy details
- Consistency details

**Visualizations:**
- Charts showing quality metrics
- Comparisons across sensors
- Trends over time (if date filtered)

## Best Practices

1. **Regular Assessment**: Check quality periodically
2. **Set Standards**: Define acceptable quality levels
3. **Monitor Trends**: Track quality over time
4. **Take Action**: Address identified issues
5. **Document Findings**: Record quality assessments
6. **Compare Periods**: Use date filters to compare

## Working with Quality Reports

### Before Assessment

- **Select Appropriate Date Range**: Focus on relevant time period
- **Ensure Data Loaded**: Verify data is available
- **Understand Context**: Know expected quality levels

### During Assessment

- **Review All Sections**: Check all quality metrics
- **Identify Issues**: Note sensors with problems
- **Compare Sensors**: Look for patterns

### After Assessment

- **Document Findings**: Record quality scores
- **Prioritize Issues**: Focus on most critical problems
- **Plan Improvements**: Address identified issues
- **Track Progress**: Re-assess after improvements

## Common Quality Issues

### Issue 1: Low Completeness

**Symptoms**: High missing percentage
**Causes**: Sensor malfunctions, data collection issues
**Actions**: Investigate sensors, check data collection system

### Issue 2: High Outlier Rate

**Symptoms**: Many outlier values
**Causes**: Sensor calibration issues, environmental factors
**Actions**: Check calibration, review operating conditions

### Issue 3: Accuracy Problems

**Symptoms**: Many threshold violations
**Causes**: Incorrect thresholds, sensor drift
**Actions**: Verify thresholds, check sensor calibration

### Issue 4: Consistency Issues

**Symptoms**: Duplicates, inconsistent timestamps
**Causes**: Data collection problems, system issues
**Actions**: Fix data collection, improve validation

## Exporting Quality Reports

**Export Options:**
- **Screenshot**: Capture quality assessment results
- **Data Export**: Export metrics as CSV (if available)
- **Report**: Generate quality report document

## Troubleshooting

### Assessment Takes Too Long

**Problem**: Quality assessment is slow
**Solutions**:
- Use date range filters
- Reduce data volume
- Check backend performance
- Verify database connection

### Unexpected Results

**Problem**: Results don't match expectations
**Solutions**:
- Verify date range
- Check sensor selection
- Review data source
- Compare with other analyses

### Missing Metrics

**Problem**: Some metrics not available
**Solutions**:
- Ensure sufficient data
- Check sensor metadata
- Verify threshold definitions
- Review error messages

## Next Steps

After quality assessment:

1. **Address Issues**: Fix identified quality problems
2. **Monitor Progress**: Re-assess after improvements
3. **Visualize Data**: Use [Data Visualization Guide](data-visualization-guide.md)
4. **Chat with Agent**: Ask [DQA Agent](dqa-agent-guide.md) about quality

## Related Documentation

- [Missing Values Guide](missing-values-guide.md) - Missing data analysis
- [Invalid Values Guide](invalid-values-guide.md) - Invalid readings analysis
- [Data Visualization Guide](data-visualization-guide.md) - Data exploration

---

*For technical details, see the [Backend API Documentation](../backend/README.md).*

