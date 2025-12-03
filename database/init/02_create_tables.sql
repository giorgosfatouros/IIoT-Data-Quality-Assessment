-- ============================================================================
-- IoT Data Quality Assessment Database Schema
-- PostgreSQL + TimescaleDB
-- ============================================================================

-- ============================================================================
-- 1. SENSOR THRESHOLDS TABLE (Migrated from tags.csv)
-- ============================================================================
-- This table stores sensor metadata and quality thresholds for validation
CREATE TABLE IF NOT EXISTS sensor_thresholds (
    tag VARCHAR(50) PRIMARY KEY,
    tag_description TEXT,
    machine_group VARCHAR(100) NOT NULL,
    low_threshold DOUBLE PRECISION,
    high_threshold DOUBLE PRECISION,
    threshold_type VARCHAR(20),  -- 'Up', 'Down', or 'Up/Down'
    aggregation_rule VARCHAR(20),  -- 'min', 'max', 'avg', 'sum'
    engineering_units VARCHAR(50),
    category VARCHAR(50),  -- Pressure, Temperature, Vibration, etc.
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for efficient lookups
CREATE INDEX idx_sensor_thresholds_machine_group ON sensor_thresholds(machine_group);
CREATE INDEX idx_sensor_thresholds_category ON sensor_thresholds(category);

-- Add comments for documentation
COMMENT ON TABLE sensor_thresholds IS 'Sensor configuration and quality thresholds for data validation';
COMMENT ON COLUMN sensor_thresholds.tag IS 'Unique sensor identifier (e.g., 22PI102)';
COMMENT ON COLUMN sensor_thresholds.threshold_type IS 'Type of threshold check: Up (max only), Down (min only), Up/Down (both)';
COMMENT ON COLUMN sensor_thresholds.aggregation_rule IS 'How to aggregate raw data: min, max, avg, or sum';

-- ============================================================================
-- 2. RAW SENSOR DATA TABLE (High-frequency sensor readings)
-- ============================================================================
-- This table stores raw sensor readings at their original frequency (e.g., 10 seconds)
-- Optimized for high-volume writes
CREATE TABLE IF NOT EXISTS raw_sensor_data (
    timestamp TIMESTAMPTZ NOT NULL,
    sensor_tag VARCHAR(50) NOT NULL,
    value DOUBLE PRECISION NOT NULL,
    machine_group VARCHAR(100) NOT NULL,
    quality_flag VARCHAR(20),  -- 'valid', 'invalid', 'missing', 'anomaly'
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Convert to TimescaleDB hypertable (partitioned by time)
SELECT create_hypertable(
    'raw_sensor_data',
    'timestamp',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '1 day'  -- Each chunk contains 1 day of data
);

-- Create indexes for efficient querying
CREATE INDEX idx_raw_sensor_data_sensor_tag ON raw_sensor_data(sensor_tag, timestamp DESC);
CREATE INDEX idx_raw_sensor_data_machine_group ON raw_sensor_data(machine_group, timestamp DESC);
CREATE INDEX idx_raw_sensor_data_quality_flag ON raw_sensor_data(quality_flag, timestamp DESC);

-- Add compression policy (compress data older than 7 days)
ALTER TABLE raw_sensor_data SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'sensor_tag,machine_group',
    timescaledb.compress_orderby = 'timestamp DESC'
);

SELECT add_compression_policy('raw_sensor_data', INTERVAL '7 days', if_not_exists => TRUE);

-- Add retention policy (keep raw data for 1 year, configurable)
-- Uncomment the following line to enable automatic data retention
-- SELECT add_retention_policy('raw_sensor_data', INTERVAL '1 year', if_not_exists => TRUE);

-- Add comments
COMMENT ON TABLE raw_sensor_data IS 'High-frequency raw sensor readings optimized for write performance';
COMMENT ON COLUMN raw_sensor_data.quality_flag IS 'Data quality status: valid, invalid (threshold violation), missing, anomaly';

-- ============================================================================
-- 3. AGGREGATED INSIGHTS TABLE (Pre-computed quality metrics)
-- ============================================================================
-- This table stores hourly aggregated quality metrics for fast API queries
-- Maintains compatibility with existing column structure (sum_, count_, etc.)
CREATE TABLE IF NOT EXISTS aggregated_insights (
    timestamp TIMESTAMPTZ NOT NULL,
    machine_group VARCHAR(100) NOT NULL,
    sensor_tag VARCHAR(50) NOT NULL,

    -- Aggregated statistics (compatible with existing structure)
    sum_value DOUBLE PRECISION,
    count_value INTEGER,
    min_value DOUBLE PRECISION,
    max_value DOUBLE PRECISION,
    avg_value DOUBLE PRECISION,
    stddev_value DOUBLE PRECISION,

    -- Quality metrics
    count_invalid INTEGER DEFAULT 0,  -- Count of threshold violations (replaces count_*_isvalid)
    count_missing INTEGER DEFAULT 0,  -- Count of missing expected readings
    count_anomaly INTEGER DEFAULT 0,  -- Count of statistical anomalies

    -- Quality scores (percentages)
    completeness_score DOUBLE PRECISION,  -- % of expected readings present
    validity_score DOUBLE PRECISION,      -- % of readings within thresholds
    anomaly_score DOUBLE PRECISION,       -- % of readings flagged as anomalies
    overall_quality_score DOUBLE PRECISION,  -- Combined quality score (0-100)

    -- Metadata
    aggregation_interval_seconds INTEGER DEFAULT 3600,  -- Duration of aggregation window
    expected_count INTEGER,  -- Expected number of readings in interval
    raw_data_ids BIGINT[],  -- References to raw data rows for drill-down (optional)

    created_at TIMESTAMPTZ DEFAULT NOW(),

    -- Prevent duplicate aggregations
    UNIQUE(timestamp, sensor_tag, machine_group, aggregation_interval_seconds)
);

-- Convert to TimescaleDB hypertable
SELECT create_hypertable(
    'aggregated_insights',
    'timestamp',
    if_not_exists => TRUE,
    chunk_time_interval => INTERVAL '1 week'  -- Each chunk contains 1 week of aggregated data
);

-- Create indexes for API query patterns
CREATE INDEX idx_aggregated_insights_sensor_tag ON aggregated_insights(sensor_tag, timestamp DESC);
CREATE INDEX idx_aggregated_insights_machine_group ON aggregated_insights(machine_group, timestamp DESC);
CREATE INDEX idx_aggregated_insights_quality_score ON aggregated_insights(overall_quality_score, timestamp DESC);
CREATE INDEX idx_aggregated_insights_timestamp ON aggregated_insights(timestamp DESC);

-- Add compression policy (compress data older than 30 days)
ALTER TABLE aggregated_insights SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'sensor_tag,machine_group',
    timescaledb.compress_orderby = 'timestamp DESC'
);

SELECT add_compression_policy('aggregated_insights', INTERVAL '30 days', if_not_exists => TRUE);

-- Add comments
COMMENT ON TABLE aggregated_insights IS 'Pre-computed hourly quality metrics for fast API queries';
COMMENT ON COLUMN aggregated_insights.completeness_score IS 'Percentage of expected readings present in window (0-100)';
COMMENT ON COLUMN aggregated_insights.validity_score IS 'Percentage of readings within threshold limits (0-100)';
COMMENT ON COLUMN aggregated_insights.overall_quality_score IS 'Weighted combined quality score (0-100)';
COMMENT ON COLUMN aggregated_insights.raw_data_ids IS 'Array of raw_sensor_data row IDs for drill-down capability';

-- ============================================================================
-- 4. CONTINUOUS AGGREGATES (Automated aggregation views)
-- ============================================================================
-- Create a continuous aggregate for hourly sensor quality metrics
-- This automatically refreshes as new data arrives
CREATE MATERIALIZED VIEW IF NOT EXISTS hourly_sensor_quality
WITH (timescaledb.continuous) AS
SELECT
    time_bucket('1 hour', timestamp) AS bucket,
    machine_group,
    sensor_tag,
    COUNT(*) AS reading_count,
    AVG(value) AS avg_value,
    MIN(value) AS min_value,
    MAX(value) AS max_value,
    STDDEV(value) AS stddev_value,
    COUNT(CASE WHEN quality_flag = 'invalid' THEN 1 END) AS invalid_count,
    COUNT(CASE WHEN quality_flag = 'missing' THEN 1 END) AS missing_count,
    COUNT(CASE WHEN quality_flag = 'anomaly' THEN 1 END) AS anomaly_count
FROM raw_sensor_data
GROUP BY bucket, machine_group, sensor_tag;

-- Add refresh policy (refresh every hour)
SELECT add_continuous_aggregate_policy('hourly_sensor_quality',
    start_offset => INTERVAL '3 hours',
    end_offset => INTERVAL '1 hour',
    schedule_interval => INTERVAL '1 hour',
    if_not_exists => TRUE
);

-- Create index on continuous aggregate
CREATE INDEX idx_hourly_sensor_quality_lookup ON hourly_sensor_quality(bucket DESC, machine_group, sensor_tag);

-- ============================================================================
-- 5. HELPER FUNCTIONS
-- ============================================================================

-- Function to calculate quality score
CREATE OR REPLACE FUNCTION calculate_quality_score(
    completeness_pct DOUBLE PRECISION,
    validity_pct DOUBLE PRECISION,
    anomaly_pct DOUBLE PRECISION
) RETURNS DOUBLE PRECISION AS $$
BEGIN
    -- Weighted composite score:
    -- Completeness: 30%, Validity: 50%, Anomaly: 20%
    RETURN (
        (completeness_pct * 0.30) +
        (validity_pct * 0.50) +
        (anomaly_pct * 0.20)
    );
END;
$$ LANGUAGE plpgsql IMMUTABLE;

COMMENT ON FUNCTION calculate_quality_score IS 'Calculate overall quality score from component scores (weighted: completeness 30%, validity 50%, anomaly 20%)';

-- Function to get sensor thresholds for a machine group
CREATE OR REPLACE FUNCTION get_sensor_thresholds(p_machine_group VARCHAR)
RETURNS TABLE (
    tag VARCHAR,
    tag_description TEXT,
    low_threshold DOUBLE PRECISION,
    high_threshold DOUBLE PRECISION,
    threshold_type VARCHAR,
    aggregation_rule VARCHAR,
    engineering_units VARCHAR,
    category VARCHAR
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        st.tag,
        st.tag_description,
        st.low_threshold,
        st.high_threshold,
        st.threshold_type,
        st.aggregation_rule,
        st.engineering_units,
        st.category
    FROM sensor_thresholds st
    WHERE st.machine_group = p_machine_group;
END;
$$ LANGUAGE plpgsql STABLE;

COMMENT ON FUNCTION get_sensor_thresholds IS 'Get all sensor thresholds for a specific machine group';




