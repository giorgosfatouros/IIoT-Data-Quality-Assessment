import { useState, useEffect } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar, ScatterChart, Scatter, Cell, ReferenceLine, ComposedChart } from 'recharts'
import { XCircle, TrendingUp, AlertCircle, Filter, ChevronDown, ChevronRight, Database, Settings, BarChart3, Activity } from 'lucide-react'
import SensorSelector from '../components/SensorSelector'

// Analytics API interfaces
interface InvalidReadingPoint {
  timestamp: string
  value: number
  alarm_count: number
}

interface TimeSeriesPoint {
  timestamp: string
  original?: number
  rolling_mean?: number
  rolling_std?: number
  alarm_count?: number
}

interface SensorMetadata {
  tag: string
  description?: string
  low_threshold?: number
  high_threshold?: number
  threshold_type?: string
  aggregation_rule?: string
  engineering_units?: string
  category?: string
}

interface SensorInvalidStats {
  sensor_name: string
  total_alarms: number
  total_readings: number
  alarm_percentage: number
  time_series: TimeSeriesPoint[]
  invalid_points: InvalidReadingPoint[]
  metadata?: SensorMetadata
}

interface InvalidValuesAnalytics {
  table_name: string
  selected_columns: string[]
  threshold: number
  total_readings: number
  total_alarms: number
  avg_alarms_per_sensor: number
  max_alarms_sensor: string
  max_alarms_count: number
  sensor_stats: SensorInvalidStats[]
  processing_info: Record<string, any>
}

interface TableData {
  columns: string[]
  rows: Record<string, any>[]
}

interface PreprocessedData {
  sensors: string[]
  readings: TableData
}

interface TagsData {
  columns: string[]
  rows: Record<string, any>[]
}

export default function InvalidValues() {
  const [tables, setTables] = useState<string[]>([])
  const [selectedTable, setSelectedTable] = useState<string>('')
  const [preprocessedData, setPreprocessedData] = useState<PreprocessedData | null>(null)
  const [tagsData, setTagsData] = useState<TagsData | null>(null)
  const [selectedColumns, setSelectedColumns] = useState<string[]>([])
  const [loading, setLoading] = useState<boolean>(false)
  const [error, setError] = useState<string>('')
  
  // Invalid values specific state
  const [analyticsData, setAnalyticsData] = useState<InvalidValuesAnalytics | null>(null)
  const [analyticsLoading, setAnalyticsLoading] = useState<boolean>(false)
  const [thresholdPercent, setThresholdPercent] = useState<number>(16.67) // Default: 16.67% alarm rate (60/360)
  const [expandedSensors, setExpandedSensors] = useState<Set<string>>(new Set())

  const API_BASE = 'http://localhost:8000'

  // Fetch tables on component mount
  useEffect(() => {
    fetchTables()
  }, [])

  // Load data when table is selected
  useEffect(() => {
    if (selectedTable) {
      loadData()
    }
  }, [selectedTable])

  const fetchTables = async () => {
    try {
      const response = await fetch(`${API_BASE}/tables`)
      if (!response.ok) throw new Error('Failed to fetch tables')
      const data = await response.json()
      // Filter to only show HOURS tables (aggregated data)
      const hoursTables = data.filter((table: string) => table.endsWith('_HOURS'))
      setTables(hoursTables)
    } catch (err: any) {
      setError(err.message)
    }
  }

  const loadData = async () => {
    setLoading(true)
    setError('')
    try {
      // Load preprocessed data
      const response = await fetch(`${API_BASE}/data/preprocessed?table=${selectedTable}`)
      if (!response.ok) throw new Error('Failed to load data')
      const data: PreprocessedData = await response.json()
      setPreprocessedData(data)
      setSelectedColumns([])
      setAnalyticsData(null)

      // Fetch tags data for sensor metadata
      try {
        const tagsResponse = await fetch(`${API_BASE}/tags?table=${selectedTable}`)
        if (tagsResponse.ok) {
          const tags: TagsData = await tagsResponse.json()
          setTagsData(tags)
        }
      } catch (err) {
        console.error('Error loading tags:', err)
        // Continue even if tags fail to load
      }
    } catch (err: any) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const getSensorInfo = (sensor: string) => {
    if (!tagsData) return null

    const tag = tagsData.rows.find(row =>
      row.tag && row.tag.toLowerCase() === sensor.toLowerCase()
    )

    if (!tag) return null

    return {
      name: sensor,
      category: tag.category || undefined,
      description: tag.description || undefined,
      unit: tag.engineering_units || tag.unit || undefined
    }
  }

  const fetchInvalidValuesAnalytics = async () => {
    if (!selectedTable) {
      setError('Please select a table first')
      return
    }

    setAnalyticsLoading(true)
    setError('')
    
    try {
      // Convert percentage to alarm count (360 readings per hour for 10-sec frequency)
      const thresholdCount = Math.round((thresholdPercent / 100) * 360)
      
      const requestBody = {
        table: selectedTable,
        columns: selectedColumns.length > 0 ? selectedColumns : null,
        threshold: thresholdCount
        // No limit - analyze complete period
      }

      const response = await fetch(`${API_BASE}/analytics/invalid-values`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestBody)
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Failed to fetch invalid values analytics')
      }

      const data: InvalidValuesAnalytics = await response.json()
      setAnalyticsData(data)
    } catch (err: any) {
      setError(err.message)
      setAnalyticsData(null)
    } finally {
      setAnalyticsLoading(false)
    }
  }

  const toggleSensorExpansion = (sensor: string) => {
    const newExpanded = new Set(expandedSensors)
    if (newExpanded.has(sensor)) {
      newExpanded.delete(sensor)
    } else {
      newExpanded.add(sensor)
    }
    setExpandedSensors(newExpanded)
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-center gap-3">
        <div className="p-3 bg-gradient-to-r from-red-600 to-orange-600 rounded-lg shadow-lg">
          <XCircle className="text-white" size={32} />
        </div>
        <div>
          <h1 className="text-4xl font-bold text-white">Invalid Values Analysis</h1>
          <p className="text-gray-400 mt-1">Analyze threshold violations and alarms from hourly aggregated sensor data</p>
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-900/50 border border-red-700 rounded-xl p-4 flex items-center gap-3 animate-in slide-in-from-top duration-300">
          <AlertCircle className="text-red-400 flex-shrink-0" size={20} />
          <span className="text-red-200">{error}</span>
        </div>
      )}

      {/* Configuration Panel */}
      <div className="bg-gradient-to-r from-gray-800 to-gray-700 rounded-xl p-6 border border-gray-600 shadow-xl">
        <div className="flex items-center gap-3 mb-6">
          <div className="p-2 bg-red-600 rounded-lg">
            <Settings className="text-white" size={24} />
          </div>
          <div>
            <h2 className="text-2xl font-semibold text-white">Analysis Configuration</h2>
            <p className="text-gray-300">Configure table selection, alarm threshold, and sensor filters</p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Table Selection */}
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                <Database className="inline mr-2" size={16} />
                Select Table
              </label>
              <select
                value={selectedTable}
                onChange={(e) => setSelectedTable(e.target.value)}
                className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
              >
                <option value="">Choose a table...</option>
                {tables.map(table => (
                  <option key={table} value={table}>{table}</option>
                ))}
              </select>
            </div>

            {/* Threshold Selection */}
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2">
                <Filter className="inline mr-2" size={16} />
                Alarm Rate Threshold
              </label>
              <div className="relative">
                <input
                  type="number"
                  min="0"
                  max="100"
                  step="0.1"
                  value={thresholdPercent}
                  onChange={(e) => setThresholdPercent(parseFloat(e.target.value) || 16.67)}
                  className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 pr-8 text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                />
                <span className="absolute right-3 top-1/2 transform -translate-y-1/2 text-gray-400">%</span>
              </div>
              <p className="text-xs text-gray-400 mt-1">Min alarm rate to display red dots</p>
            </div>

            {/* Analyze Button */}
            <div>
              <label className="block text-sm font-medium text-gray-300 mb-2 opacity-0">
                Action
              </label>
              <button
                onClick={fetchInvalidValuesAnalytics}
                disabled={!selectedTable || analyticsLoading}
                className="w-full bg-gradient-to-r from-red-600 to-orange-600 hover:from-red-700 hover:to-orange-700 disabled:from-gray-600 disabled:to-gray-700 text-white font-semibold py-2 px-4 rounded-lg transition-all duration-200 flex items-center justify-center"
              >
                {analyticsLoading ? (
                  <>
                    <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
                    Analyzing...
                  </>
                ) : (
                  <>
                    <TrendingUp className="mr-2" size={18} />
                    Analyze
                  </>
                )}
              </button>
            </div>
          </div>

        {/* Column Selection */}
        {preprocessedData && (
          <SensorSelector
            sensors={preprocessedData.sensors}
            selectedSensors={selectedColumns}
            onSelectionChange={setSelectedColumns}
            getSensorInfo={getSensorInfo}
            accentColor="red"
          />
        )}
        </div>

      {/* Analytics Results */}
      {analyticsData && (
        <div className="space-y-6">
          {/* Overall Statistics */}
          <div className="bg-gradient-to-r from-gray-800 to-gray-700 rounded-xl p-6 border border-gray-600 shadow-xl">
            <div className="flex items-center gap-3 mb-6">
              <div className="p-2 bg-red-600 rounded-lg">
                <AlertCircle className="text-white" size={24} />
              </div>
              <div>
                <h2 className="text-2xl font-semibold text-white">Overall Statistics</h2>
                <p className="text-gray-300">Summary of alarm data across all analyzed sensors</p>
              </div>
            </div>
              
              {analyticsData.total_alarms === 0 ? (
                <div className="text-center py-8">
                  <XCircle className="mx-auto text-green-400 mb-3" size={48} />
                  <p className="text-xl text-green-400 font-semibold">No alarms detected!</p>
                  <p className="text-gray-400 mt-2">All sensors are operating normally with no invalid readings above the threshold.</p>
                </div>
              ) : (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="bg-gray-900/50 rounded-lg p-4 border border-gray-700">
                    <p className="text-gray-400 text-sm mb-1">Total Readings</p>
                    <p className="text-2xl font-bold text-white">{analyticsData.total_readings.toLocaleString()}</p>
                  </div>
                  <div className="bg-red-900/30 rounded-lg p-4 border border-red-700">
                    <p className="text-gray-400 text-sm mb-1">Total Alarms</p>
                    <p className="text-2xl font-bold text-red-400">{analyticsData.total_alarms.toLocaleString()}</p>
                  </div>
                  <div className="bg-orange-900/30 rounded-lg p-4 border border-orange-700">
                    <p className="text-gray-400 text-sm mb-1">Avg Alarms per Sensor</p>
                    <p className="text-2xl font-bold text-orange-400">{analyticsData.avg_alarms_per_sensor.toFixed(1)}</p>
                  </div>
                  <div className="bg-yellow-900/30 rounded-lg p-4 border border-yellow-700">
                    <p className="text-gray-400 text-sm mb-1">Most Affected Sensor</p>
                    <p className="text-lg font-bold text-yellow-400">{analyticsData.max_alarms_sensor}</p>
                    <p className="text-xs text-gray-400">{analyticsData.max_alarms_count.toLocaleString()} alarms</p>
                  </div>
                </div>
              )}

              <div className="mt-4 text-xs text-gray-400">
                <p>Display threshold: {thresholdPercent}% alarm rate ({Math.round((thresholdPercent / 100) * 360)} alarms/hour)</p>
                <p>Sensors with alarms: {analyticsData.sensor_stats.length}</p>
              </div>
            </div>

          {/* Sensors with Alarms Bar Chart */}
          {analyticsData.sensor_stats.length > 0 && (
            <div className="bg-gradient-to-r from-gray-800 to-gray-700 rounded-xl p-6 border border-gray-600 shadow-xl">
              <div className="flex items-center gap-3 mb-6">
                <div className="p-2 bg-orange-600 rounded-lg">
                  <BarChart3 className="text-white" size={24} />
                </div>
                <div>
                  <h2 className="text-2xl font-semibold text-white">Alarms by Sensor</h2>
                  <p className="text-gray-300">Total alarm count distribution across sensors</p>
                </div>
              </div>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={analyticsData.sensor_stats}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis 
                      dataKey="sensor_name" 
                      stroke="#9CA3AF" 
                      angle={-45}
                      textAnchor="end"
                      height={100}
                    />
                    <YAxis stroke="#9CA3AF" />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151' }}
                      labelStyle={{ color: '#F3F4F6' }}
                    />
                    <Legend />
                    <Bar dataKey="total_alarms" fill="#EF4444" name="Total Alarms" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}

          {/* Alarm Percentage Chart */}
          {analyticsData.sensor_stats.length > 0 && (
            <div className="bg-gradient-to-r from-gray-800 to-gray-700 rounded-xl p-6 border border-gray-600 shadow-xl">
              <div className="flex items-center gap-3 mb-6">
                <div className="p-2 bg-yellow-600 rounded-lg">
                  <TrendingUp className="text-white" size={24} />
                </div>
                <div>
                  <h2 className="text-2xl font-semibold text-white">Alarm Percentage by Sensor</h2>
                  <p className="text-gray-300">Alarm rate percentage distribution across sensors</p>
                </div>
              </div>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={analyticsData.sensor_stats}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                    <XAxis 
                      dataKey="sensor_name" 
                      stroke="#9CA3AF" 
                      angle={-45}
                      textAnchor="end"
                      height={100}
                    />
                    <YAxis stroke="#9CA3AF" />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151' }}
                      labelStyle={{ color: '#F3F4F6' }}
                      formatter={(value: number) => `${value.toFixed(2)}%`}
                    />
                    <Legend />
                    <Bar dataKey="alarm_percentage" fill="#F59E0B" name="Alarm %" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}

          {/* Detailed Sensor Analysis */}
          {analyticsData.sensor_stats.length > 0 && (
            <div className="bg-gradient-to-r from-gray-800 to-gray-700 rounded-xl p-6 border border-gray-600 shadow-xl">
              <div className="flex items-center gap-3 mb-6">
                <div className="p-2 bg-indigo-600 rounded-lg">
                  <Activity className="text-white" size={24} />
                </div>
                <div>
                  <h2 className="text-2xl font-semibold text-white">Detailed Sensor Analysis</h2>
                  <p className="text-gray-300">Individual sensor time series with alarm indicators</p>
                </div>
              </div>
                <div className="space-y-4">
                  {analyticsData.sensor_stats.map((sensor) => (
                    <div key={sensor.sensor_name} className="bg-gray-900/50 rounded-lg border border-gray-700 overflow-hidden">
                      <div
                        className="flex items-center justify-between p-4 cursor-pointer hover:bg-gray-800/50 transition-colors"
                        onClick={() => toggleSensorExpansion(sensor.sensor_name)}
                      >
                        <div className="flex items-center space-x-3">
                          {expandedSensors.has(sensor.sensor_name) ? (
                            <ChevronDown className="text-gray-400" size={20} />
                          ) : (
                            <ChevronRight className="text-gray-400" size={20} />
                          )}
                          <h4 className="text-lg font-semibold text-white">{sensor.sensor_name}</h4>
                        </div>
                        <div className="flex items-center space-x-4 text-sm">
                          <span className="text-red-400 font-semibold">
                            {sensor.total_alarms.toLocaleString()} alarms
                          </span>
                          <span className="text-orange-400">
                            {sensor.alarm_percentage.toFixed(2)}%
                          </span>
                        </div>
                      </div>

                      {expandedSensors.has(sensor.sensor_name) && (
                        <div className="p-4 border-t border-gray-700 space-y-4">
                          {/* Debug Info */}
                          {sensor.invalid_points.length > 0 && (
                            <div className="bg-blue-900/30 rounded p-2 text-xs text-blue-300">
                              <p>Debug: {sensor.invalid_points.length} invalid points found (≥{thresholdPercent}% alarm rate)</p>
                              <p>First point: {sensor.invalid_points[0]?.timestamp} - {sensor.invalid_points[0]?.alarm_count} alarms ({((sensor.invalid_points[0]?.alarm_count / 360) * 100).toFixed(1)}%)</p>
                            </div>
                          )}
                          
                          {/* Sensor Metadata */}
                          {sensor.metadata && (
                            <div className="bg-gray-800/50 rounded p-3 text-sm">
                              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                                {sensor.metadata.description && (
                                  <div>
                                    <p className="text-gray-400 text-xs">Description</p>
                                    <p className="text-white font-medium">{sensor.metadata.description}</p>
                                  </div>
                                )}
                                {sensor.metadata.engineering_units && (
                                  <div>
                                    <p className="text-gray-400 text-xs">Units</p>
                                    <p className="text-white font-medium">{sensor.metadata.engineering_units}</p>
                                  </div>
                                )}
                                {sensor.metadata.category && (
                                  <div>
                                    <p className="text-gray-400 text-xs">Category</p>
                                    <p className="text-white font-medium">{sensor.metadata.category}</p>
                                  </div>
                                )}
                                {sensor.metadata.threshold_type && (
                                  <div>
                                    <p className="text-gray-400 text-xs">Threshold Type</p>
                                    <p className="text-white font-medium">{sensor.metadata.threshold_type}</p>
                                  </div>
                                )}
                              </div>
                            </div>
                          )}

                          {/* Combined Time Series with Invalid Points and Threshold Lines */}
                          <div>
                            <h5 className="text-md font-semibold text-white mb-3">
                              Time Series with Invalid Readings
                              {sensor.metadata?.engineering_units && (
                                <span className="text-sm text-gray-400 ml-2">
                                  ({sensor.metadata.engineering_units})
                                </span>
                              )}
                            </h5>
                            <ResponsiveContainer width="100%" height={350}>
                              <ComposedChart data={sensor.time_series} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                                <XAxis 
                                  dataKey="timestamp" 
                                  stroke="#9CA3AF"
                                  tickFormatter={(value) => new Date(value).toLocaleDateString()}
                                />
                                <YAxis stroke="#9CA3AF" />
                                <Tooltip 
                                  contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151' }}
                                  labelStyle={{ color: '#F3F4F6' }}
                                  labelFormatter={(value) => new Date(value).toLocaleString()}
                                />
                                <Legend />
                                
                                {/* Threshold Reference Lines */}
                                {sensor.metadata?.high_threshold !== undefined && sensor.metadata.high_threshold !== null && (
                                  <ReferenceLine 
                                    y={sensor.metadata.high_threshold} 
                                    stroke="#F59E0B" 
                                    strokeWidth={2}
                                    strokeDasharray="5 5"
                                    label={{ 
                                      value: `High: ${sensor.metadata.high_threshold}`, 
                                      position: 'right',
                                      fill: '#F59E0B',
                                      fontSize: 12
                                    }}
                                  />
                                )}
                                
                                {sensor.metadata?.low_threshold !== undefined && sensor.metadata.low_threshold !== null && (
                                  <ReferenceLine 
                                    y={sensor.metadata.low_threshold} 
                                    stroke="#F59E0B" 
                                    strokeWidth={2}
                                    strokeDasharray="5 5"
                                    label={{ 
                                      value: `Low: ${sensor.metadata.low_threshold}`, 
                                      position: 'right',
                                      fill: '#F59E0B',
                                      fontSize: 12
                                    }}
                                  />
                                )}
                                
                                {/* Time Series Line with conditional red dots for alarms */}
                                <Line 
                                  type="monotone" 
                                  dataKey="original" 
                                  stroke="#3B82F6" 
                                  name="Mean Value"
                                  strokeWidth={2}
                                  dot={(props: any) => {
                                    const { cx, cy, payload } = props;
                                    // Convert percentage to alarm count threshold
                                    const thresholdCount = Math.round((thresholdPercent / 100) * 360);
                                    // Show red dot if alarm_count >= threshold
                                    if (payload.alarm_count && payload.alarm_count >= thresholdCount) {
                                      const color = payload.alarm_count >= thresholdCount * 2 ? '#DC2626' : '#EF4444';
                                      return (
                                        <circle
                                          cx={cx}
                                          cy={cy}
                                          r={5}
                                          fill={color}
                                          stroke="#FFF"
                                          strokeWidth={1}
                                        />
                                      );
                                    }
                                    return null; // No dot for normal points
                                  }}
                                />
                              </ComposedChart>
                            </ResponsiveContainer>
                            
                            {/* Threshold Legend */}
                            <div className="flex items-center justify-center gap-6 mt-3 text-xs">
                              <div className="flex items-center gap-2">
                                <div className="w-8 h-0.5 bg-blue-500"></div>
                                <span className="text-gray-300">Mean Value</span>
                              </div>
                              <div className="flex items-center gap-2">
                                <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                                <span className="text-gray-300">≥{thresholdPercent}% Alarm Rate</span>
                              </div>
                              <div className="flex items-center gap-2">
                                <div className="w-3 h-3 bg-red-900 rounded-full"></div>
                                <span className="text-gray-300">≥{(thresholdPercent * 2).toFixed(1)}% (Critical)</span>
                              </div>
                              {(sensor.metadata?.high_threshold !== undefined || sensor.metadata?.low_threshold !== undefined) && (
                                <div className="flex items-center gap-2">
                                  <div className="w-8 h-0.5 border-t-2 border-dashed border-orange-500"></div>
                                  <span className="text-gray-300">Threshold Limits</span>
                                </div>
                              )}
                            </div>
                            
                            {/* Threshold Values Display */}
                            {(sensor.metadata?.low_threshold !== undefined || sensor.metadata?.high_threshold !== undefined) && (
                              <div className="mt-2 text-xs text-center text-gray-400">
                                {sensor.metadata.low_threshold !== undefined && sensor.metadata.low_threshold !== null && (
                                  <span className="mr-4">
                                    Low: <span className="text-orange-400 font-semibold">{sensor.metadata.low_threshold}</span>
                                  </span>
                                )}
                                {sensor.metadata.high_threshold !== undefined && sensor.metadata.high_threshold !== null && (
                                  <span>
                                    High: <span className="text-orange-400 font-semibold">{sensor.metadata.high_threshold}</span>
                                  </span>
                                )}
                                {sensor.metadata.engineering_units && (
                                  <span className="ml-2">({sensor.metadata.engineering_units})</span>
                                )}
                              </div>
                            )}
                            
                            <p className="text-xs text-gray-400 mt-2 text-center">
                              Showing {sensor.invalid_points.length} hours with ≥{thresholdPercent}% alarm rate (red dots on time series)
                            </p>
                          </div>

                          {/* Statistics Summary */}
                          <div className="grid grid-cols-3 gap-3 pt-3 border-t border-gray-700">
                            <div className="bg-gray-800/50 rounded p-3">
                              <p className="text-xs text-gray-400">Total Readings</p>
                              <p className="text-lg font-semibold text-white">{sensor.total_readings.toLocaleString()}</p>
                            </div>
                            <div className="bg-gray-800/50 rounded p-3">
                              <p className="text-xs text-gray-400">Total Alarms</p>
                              <p className="text-lg font-semibold text-red-400">{sensor.total_alarms.toLocaleString()}</p>
                            </div>
                            <div className="bg-gray-800/50 rounded p-3">
                              <p className="text-xs text-gray-400">Alarm Rate</p>
                              <p className="text-lg font-semibold text-orange-400">{sensor.alarm_percentage.toFixed(3)}%</p>
                            </div>
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

      {/* Loading State */}
      {loading && !preprocessedData && (
        <div className="text-center py-12">
          <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-red-500 mx-auto mb-4"></div>
          <h3 className="text-xl font-semibold text-white mb-2">Loading Data</h3>
          <p className="text-gray-300">Please wait while we fetch your sensor data...</p>
        </div>
      )}

      {/* Empty State */}
      {!loading && !analyticsData && preprocessedData && (
        <div className="text-center py-12">
          <AlertCircle className="mx-auto text-gray-500 mb-4" size={64} />
          <h3 className="text-xl font-semibold text-gray-300 mb-2">Get Started</h3>
          <p className="text-gray-400">Select a table and click "Analyze Invalid Values" to begin</p>
        </div>
      )}
    </div>
  )
}