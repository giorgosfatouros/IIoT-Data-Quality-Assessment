# Methodology and Innovation

## Introduction

Industrial Internet of Things (IIoT) systems generate high-frequency sensor data at rates of one reading every few seconds. Analyzing months or years of such data presents significant computational challenges when processing millions of raw data points for each query. This service addresses this challenge through a two-tier architecture that combines raw data preservation with pre-computed aggregated insights, enabling efficient analysis of large-scale time-series data.

## Methodology

### Two-Tier Architecture

The service employs a two-tier data architecture optimized for high-frequency time-series data:

**Tier 1: Raw Data Storage**
- Raw sensor readings are stored at their original collection frequency (typically 10-second intervals)
- Data is stored in TimescaleDB hypertables, which partition data by time for efficient write and query performance
- Each reading is automatically annotated with a quality flag during import (valid, invalid, missing, or anomaly)
- Raw data is preserved for drill-down analysis and historical reference

**Tier 2: Aggregated Insights**
- A background worker service continuously aggregates raw data into time windows (default: hourly intervals)
- Pre-computed quality metrics are calculated and stored for each aggregation window:
  - Statistical summaries (min, max, mean, standard deviation)
  - Quality scores (completeness, validity, anomaly, overall quality)
  - Quality issue counts (invalid readings, missing readings, anomalies)
- Aggregated data enables fast queries without processing millions of raw points

### Online Aggregation Process

The aggregation process runs continuously in the background:

1. **Data Import**: Raw sensor data is imported with automatic quality flag annotation based on threshold validation
2. **Background Processing**: A worker service identifies pending aggregation windows and processes them incrementally
3. **Quality Assessment**: For each window, the system:
   - Validates readings against sensor-specific thresholds
   - Detects statistical anomalies using Z-score analysis
   - Calculates completeness based on expected vs. actual readings
   - Computes composite quality scores
4. **Storage**: Aggregated insights are stored in a separate hypertable optimized for read queries

This approach ensures that quality metrics are available immediately after data import, without requiring expensive on-the-fly calculations.

## Value Proposition

### Efficient Analysis of Large Datasets

The two-tier architecture enables efficient analysis of high-frequency data:

- **Query Performance**: Most analytical queries operate on aggregated data (hourly windows) rather than raw 10-second intervals, reducing data volume by 360x
- **Scalability**: The system can handle months or years of data without performance degradation, as queries scale with the number of aggregation windows rather than raw data points
- **Real-Time Assessment**: Quality metrics are computed as data arrives, enabling immediate insights without waiting for batch processing

### Practical Benefits

- **Fast Dashboard Loading**: Aggregated data enables responsive visualizations and analytics
- **Reduced Computational Load**: Pre-computed metrics eliminate redundant calculations
- **Preserved Raw Data Access**: Raw data remains available for detailed analysis when needed
- **Automatic Quality Monitoring**: Quality flags and scores are automatically maintained as new data arrives

## Innovation Highlights

### Automatic Quality Annotation

During data import, each sensor reading is automatically annotated with a quality flag:

- **Valid**: Reading is within acceptable threshold ranges
- **Invalid**: Reading violates sensor-specific thresholds (low/high limits)
- **Missing**: Expected reading is absent (detected during aggregation)
- **Anomaly**: Reading is statistically anomalous (detected via Z-score analysis)

This annotation occurs at import time, enabling immediate quality assessment without post-processing.

### Continuous Background Aggregation

The aggregation worker runs continuously, processing new data windows as they become available:

- **Incremental Processing**: Only pending windows are processed, avoiding redundant work
- **Non-Blocking**: Aggregation occurs in the background without impacting user queries
- **Automatic Updates**: Quality metrics are refreshed as new data arrives

### Pre-Computed Quality Metrics

Quality scores are calculated and stored for each aggregation window:

- **Completeness Score**: Percentage of expected readings present (0-100%)
- **Validity Score**: Percentage of readings within threshold limits (0-100%)
- **Anomaly Score**: Percentage of readings that are not statistical anomalies (0-100%)
- **Overall Quality Score**: Weighted composite score (Completeness 30%, Validity 50%, Anomaly 20%)

These pre-computed metrics enable instant quality assessment without recalculating from raw data.

## Technical Approach

### TimescaleDB Optimization

The database layer leverages TimescaleDB features for time-series optimization:

- **Hypertables**: Automatic partitioning by time (1-day chunks for raw data, 1-week chunks for aggregated data)
- **Compression**: Automatic compression of older data (7 days for raw data, 30 days for aggregated data)
- **Indexes**: Optimized indexes on sensor tags, machine groups, timestamps, and quality flags
- **Continuous Aggregates**: Materialized views for common query patterns (optional, for future use)

### Aggregation Configuration

- **Aggregation Interval**: Configurable time window (default: 3600 seconds / 1 hour)
- **Original Frequency**: Expected data collection frequency (default: 10 seconds)
- **Expected Readings**: Calculated as aggregation_interval / original_frequency (e.g., 360 readings per hour)

### Quality Assessment Engine

The quality assessment engine performs two types of validation:

1. **Threshold Validation**: Compares readings against sensor-specific thresholds (low/high limits, threshold types: Up, Down, Up/Down)
2. **Statistical Anomaly Detection**: Uses Z-score analysis (default threshold: 2.0 standard deviations) to identify outliers

Both validations occur during aggregation, with results stored in the aggregated insights table.

## Service Capabilities

The service provides a comprehensive suite of tools for IIoT data quality assessment:

### Data Import & Loading
- CSV file upload with automatic validation
- Automatic machine type detection (KT2201, K3301, K5700, or auto-detect)
- Sensor selection during import (filter to specific sensors)
- Direct import to TimescaleDB with quality flag annotation
- Progress tracking for large file imports

### Data Visualization
Interactive charts and analytics powered by aggregated data:
- **Time Series Analysis**: Trend visualization with rolling statistics
- **Correlation Analysis**: Correlation matrix for sensor relationships
- **Distribution Analysis**: Histograms and density plots
- **Box Plot Analysis**: Quartile and outlier visualization
- **Seasonal Decomposition**: Trend, seasonal, and residual components
- **Anomaly Detection**: Statistical anomaly identification and visualization

### Data Quality Assessment
Comprehensive quality metrics and analysis:
- **Completeness Metrics**: Overall and per-sensor completeness scores
- **Accuracy Checks**: Threshold violation detection and reporting
- **Consistency Validation**: Duplicate detection and timestamp validation
- **Outlier Detection**: Statistical outlier identification using IQR method
- **Correlation Analysis**: Strong correlation identification between sensors

### Missing Values Analysis
- Detection of missing expected readings
- Per-sensor missing value statistics
- Time-based missing value patterns
- Completeness scoring and reporting

### Invalid Values Analysis
- Identification of threshold violations
- Per-sensor invalid reading counts and percentages
- Threshold type and limit information
- Quality flag-based filtering and analysis

### DQA Agent
AI-powered chat assistant for natural language interaction:
- Query sensor data quality metrics in natural language
- Ask about threshold violations and alarm rates
- Request data completeness and missing value analysis
- Get explanations of sensor thresholds and configurations
- Generate comprehensive data quality reports
- Analyze specific sensors or machine groups

The agent operates on aggregated data, enabling fast responses to complex queries about data quality across large time ranges.

---

*This methodology enables efficient analysis of high-frequency IIoT data by combining raw data preservation with pre-computed aggregated insights, providing both detailed access and fast analytical queries.*

