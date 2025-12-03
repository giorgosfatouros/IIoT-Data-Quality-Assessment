import { useState, useEffect } from 'react'
import { Shield, Database, TrendingUp, BarChart3, AlertCircle, CheckCircle, XCircle, Activity, AlertTriangle, Calendar, Filter } from 'lucide-react'
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts'
import Footer from '../components/Footer'

// Interfaces
interface SensorDataPoints {
  sensor_name: string
  data_points: number
  missing_percentage: number
}

interface GeneralInfo {
  date_range_start: string | null
  date_range_end: string | null
  total_data_points: number
  total_missing_values: number
  missing_percentage: number
  num_sensors: number
  sensor_data_points: SensorDataPoints[]
}

interface DescriptiveStats {
  sensor_name: string
  count: number
  mean: number
  std: number
  min: number
  q25: number
  q50: number
  q75: number
  max: number
}

interface ConsistencyCheck {
  has_duplicates: boolean
  duplicate_count: number
  duplicate_percentage: number
  timestamps_consistent: boolean
}

interface CompletenessCheck {
  overall_completeness: number
  completeness_threshold: number
  incomplete_sensors: Array<{sensor_name: string, completeness: number}>
}

interface OutlierInfo {
  sensor_name: string
  outlier_percentage: number
}

interface AccuracyIssue {
  sensor_name: string
  issues_percentage: number
  threshold_type: string | null
  low_threshold: number | null
  high_threshold: number | null
}

interface CorrelationPair {
  sensor_a: string
  sensor_b: string
  correlation: number
}

interface DataQualityAnalytics {
  table_name: string
  general_info: GeneralInfo
  descriptive_stats: DescriptiveStats[]
  consistency_check: ConsistencyCheck
  completeness_check: CompletenessCheck
  outliers: OutlierInfo[]
  accuracy_issues: AccuracyIssue[]
  strong_correlations: CorrelationPair[]
  correlation_matrix: {
    columns: string[]
    data: number[][]
  } | null
}

export default function DataQuality() {
  const [tables, setTables] = useState<string[]>([])
  const [selectedTable, setSelectedTable] = useState<string>('')
  const [analyticsData, setAnalyticsData] = useState<DataQualityAnalytics | null>(null)
  const [loading, setLoading] = useState<boolean>(false)
  const [analyticsLoading, setAnalyticsLoading] = useState<boolean>(false)
  const [error, setError] = useState<string>('')
  
  // Date range filtering
  const [dateFrom, setDateFrom] = useState<string>('')
  const [dateTo, setDateTo] = useState<string>('')
  const [availableDateRange, setAvailableDateRange] = useState<{min: string, max: string} | null>(null)

  const API_BASE = 'http://localhost:8000'

  useEffect(() => {
    fetchTables()
  }, [])

  useEffect(() => {
    if (selectedTable) {
      setAnalyticsData(null)
      setDateFrom('')
      setDateTo('')
      loadAvailableDateRange()
    }
  }, [selectedTable])

  const loadAvailableDateRange = async () => {
    if (!selectedTable) return
    
    try {
      // Fetch a small sample to get the date range
      const response = await fetch(`${API_BASE}/data?table=${selectedTable}&limit=1000`)
      if (!response.ok) return
      
      const data = await response.json()
      if (data.rows && data.rows.length > 0) {
        // Extract timestamps
        const timestamps = data.rows
          .map((row: any) => row.TIMESTAMP || row.timestamp)
          .filter((ts: any) => ts)
          .map((ts: any) => new Date(ts))
          .sort((a: Date, b: Date) => a.getTime() - b.getTime())
        
        if (timestamps.length > 0) {
          const minDate = timestamps[0]
          const maxDate = timestamps[timestamps.length - 1]
          
          const formatDate = (date: Date) => {
            const year = date.getFullYear()
            const month = String(date.getMonth() + 1).padStart(2, '0')
            const day = String(date.getDate()).padStart(2, '0')
            return `${year}-${month}-${day}`
          }
          
          const minFormatted = formatDate(minDate)
          const maxFormatted = formatDate(maxDate)
          
          setAvailableDateRange({
            min: minFormatted,
            max: maxFormatted
          })
          
          // Set default to full range
          setDateFrom(minFormatted)
          setDateTo(maxFormatted)
        }
      }
    } catch (err) {
      console.error('Error loading date range:', err)
    }
  }

  const fetchTables = async () => {
    try {
      const response = await fetch(`${API_BASE}/tables`)
      if (!response.ok) throw new Error('Failed to fetch tables')
      const data = await response.json()
      // Use all machine groups from aggregated_insights (no filtering needed)
      // The /tables endpoint already returns distinct machine_group values
      setTables(data)
      if (data.length > 0 && !selectedTable) {
        setSelectedTable(data[0])
      }
    } catch (err: any) {
      setError(err.message)
    }
  }

  const fetchDataQualityAnalytics = async () => {
    if (!selectedTable) return

    setAnalyticsLoading(true)
    setError('')
    
    try {
      const requestBody: any = {
        table: selectedTable,
        limit: null,  // Changed to null to get all data in date range
        completeness_threshold: 90.0,
        correlation_threshold: 0.7
      }

      // Add date filters if provided
      if (dateFrom) {
        requestBody.date_from = dateFrom
      }
      if (dateTo) {
        requestBody.date_to = dateTo
      }

      console.log('Sending data quality request:', requestBody)

      const response = await fetch(`${API_BASE}/analytics/data-quality`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestBody)
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Failed to fetch data quality analytics')
      }

      const data: DataQualityAnalytics = await response.json()
      console.log('Received data quality response:', data)
      setAnalyticsData(data)
    } catch (err: any) {
      setError(err.message)
      setAnalyticsData(null)
    } finally {
      setAnalyticsLoading(false)
    }
  }

  const getOutliersChartData = () => {
    if (!analyticsData) return []
    return analyticsData.outliers
      .filter(o => o.outlier_percentage > 0)
      .sort((a, b) => b.outlier_percentage - a.outlier_percentage)
      .slice(0, 20)
      .map(o => ({
        sensor: o.sensor_name,
        percentage: parseFloat(o.outlier_percentage.toFixed(2))
      }))
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-center gap-3">
        <div className="p-3 bg-gradient-to-r from-green-600 to-emerald-600 rounded-lg shadow-lg">
          <Shield className="text-white" size={32} />
        </div>
        <div>
          <h1 className="text-4xl font-bold text-white">Data Quality Assessment</h1>
          <p className="text-gray-400 mt-1">Comprehensive data quality metrics and analysis for your sensor data</p>
        </div>
      </div>

      {/* Error Display */}
      {error && (
        <div className="bg-red-900/50 border border-red-700 rounded-xl p-4 flex items-center gap-3">
          <AlertCircle className="text-red-400 flex-shrink-0" size={20} />
          <span className="text-red-200">{error}</span>
        </div>
      )}

      {/* Configuration Panel */}
      <div className="bg-gradient-to-r from-gray-800 to-gray-700 rounded-xl p-6 border border-gray-600 shadow-xl">
        <div className="flex items-center gap-3 mb-6">
          <div className="p-2 bg-green-600 rounded-lg">
            <Database className="text-white" size={24} />
          </div>
          <div>
            <h2 className="text-2xl font-semibold text-white">Analysis Configuration</h2>
            <p className="text-gray-300">Select a table to analyze data quality</p>
          </div>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Table Selection */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2">
              <Database className="inline mr-2" size={16} />
              Select Table
            </label>
            <select
              value={selectedTable}
              onChange={(e) => setSelectedTable(e.target.value)}
              className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-white focus:ring-2 focus:ring-green-500 focus:border-transparent"
            >
              <option value="">Choose a table...</option>
              {tables.map(table => (
                <option key={table} value={table}>{table}</option>
              ))}
            </select>
          </div>

          {/* Analyze Button */}
          <div>
            <label className="block text-sm font-medium text-gray-300 mb-2 opacity-0">
              Action
            </label>
            <button
              onClick={fetchDataQualityAnalytics}
              disabled={!selectedTable || analyticsLoading}
              className="w-full bg-gradient-to-r from-green-600 to-emerald-600 hover:from-green-700 hover:to-emerald-700 disabled:from-gray-600 disabled:to-gray-700 text-white font-semibold py-2 px-4 rounded-lg transition-all duration-200 flex items-center justify-center"
            >
              {analyticsLoading ? (
                <>
                  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white mr-2"></div>
                  Analyzing...
                </>
              ) : (
                <>
                  <TrendingUp className="mr-2" size={18} />
                  Analyze Quality
                </>
              )}
            </button>
          </div>
        </div>
      </div>

      {/* Date Filtering Options */}
      {selectedTable && (
        <div className="bg-gradient-to-r from-gray-800 to-gray-700 rounded-xl p-6 border border-gray-600 shadow-xl">
          <div className="flex items-center gap-3 mb-6">
            <div className="p-2 bg-orange-600 rounded-lg">
              <Filter className="text-white" size={24} />
            </div>
            <div>
              <h2 className="text-2xl font-semibold text-white">Data Filtering</h2>
              <p className="text-gray-300">Filter your data by date range (optional)</p>
            </div>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Date From */}
            <div className="space-y-2">
              <label className="block text-sm font-medium text-gray-300">
                <Calendar className="inline mr-2" size={16} />
                Start Date
              </label>
              <input
                type="date"
                value={dateFrom}
                onChange={(e) => setDateFrom(e.target.value)}
                min={availableDateRange?.min}
                max={availableDateRange?.max}
                className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-3 text-white focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-transparent transition-all"
              />
              <p className="text-xs text-gray-400">
                Optional: Filter data from this date
                {availableDateRange && (
                  <span className="ml-2 text-blue-400">
                    (Available: {availableDateRange.min} to {availableDateRange.max})
                  </span>
                )}
              </p>
            </div>

            {/* Date To */}
            <div className="space-y-2">
              <label className="block text-sm font-medium text-gray-300">
                <Calendar className="inline mr-2" size={16} />
                End Date
              </label>
              <input
                type="date"
                value={dateTo}
                onChange={(e) => setDateTo(e.target.value)}
                min={availableDateRange?.min}
                max={availableDateRange?.max}
                className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-3 text-white focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-transparent transition-all"
              />
              <p className="text-xs text-gray-400">
                Optional: Filter data until this date
                {availableDateRange && (
                  <span className="ml-2 text-blue-400">
                    (Available: {availableDateRange.min} to {availableDateRange.max})
                  </span>
                )}
              </p>
            </div>
          </div>

          {/* Filter Summary */}
          {(dateFrom || dateTo) && (
            <div className="mt-6 bg-orange-900/20 border border-orange-600/30 rounded-lg p-4">
              <div className="flex items-center gap-2 mb-2">
                <Filter className="text-orange-400" size={16} />
                <span className="text-orange-300 font-medium">Active Filters</span>
              </div>
              <div className="flex flex-wrap gap-2 text-sm">
                {dateFrom && (
                  <span className="bg-orange-600 text-white px-2 py-1 rounded-full">
                    From: {dateFrom}
                  </span>
                )}
                {dateTo && (
                  <span className="bg-orange-600 text-white px-2 py-1 rounded-full">
                    To: {dateTo}
                  </span>
                )}
              </div>
              <button
                onClick={() => {
                  // Reset to full available range
                  if (availableDateRange) {
                    setDateFrom(availableDateRange.min)
                    setDateTo(availableDateRange.max)
                  }
                }}
                className="mt-3 text-orange-300 hover:text-orange-200 text-sm underline"
              >
                Reset to full range
              </button>
            </div>
          )}
        </div>
      )}

      {/* Analytics Results */}
      {analyticsData && (
        <>
          {/* General Information */}
          <div className="bg-gradient-to-r from-gray-800 to-gray-700 rounded-xl p-6 border border-gray-600 shadow-xl">
            <div className="flex items-center gap-3 mb-6">
              <div className="p-2 bg-blue-600 rounded-lg">
                <Database className="text-white" size={24} />
              </div>
              <div>
                <h2 className="text-2xl font-semibold text-white">General Information</h2>
                <p className="text-gray-300">
                  Overview of dataset characteristics
                  {(dateFrom || dateTo) && (
                    <span className="ml-2 text-orange-400 text-sm">
                      (Date filtered)
                    </span>
                  )}
                </p>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="bg-gray-900 rounded-lg p-4">
                <div className="text-gray-400 text-sm mb-1">Date Range</div>
                <div className="text-lg font-bold text-white">
                  {analyticsData.general_info.date_range_start && analyticsData.general_info.date_range_end
                    ? `${new Date(analyticsData.general_info.date_range_start).toLocaleDateString()} - ${new Date(analyticsData.general_info.date_range_end).toLocaleDateString()}`
                    : 'N/A'}
                </div>
                {(dateFrom || dateTo) && (
                  <div className="text-xs text-orange-400 mt-1">
                    Filtered
                  </div>
                )}
              </div>
              <div className="bg-gray-900 rounded-lg p-4">
                <div className="text-gray-400 text-sm mb-1">Total Data Points</div>
                <div className="text-2xl font-bold text-blue-400">
                  {analyticsData.general_info.total_data_points.toLocaleString()}
                </div>
              </div>
              <div className="bg-gray-900 rounded-lg p-4">
                <div className="text-gray-400 text-sm mb-1">Missing Values</div>
                <div className="text-2xl font-bold text-orange-400">
                  {analyticsData.general_info.total_missing_values.toLocaleString()}
                </div>
                <div className="text-xs text-gray-500">
                  {analyticsData.general_info.missing_percentage.toFixed(2)}%
                </div>
              </div>
              <div className="bg-gray-900 rounded-lg p-4">
                <div className="text-gray-400 text-sm mb-1">Number of Sensors</div>
                <div className="text-2xl font-bold text-green-400">
                  {analyticsData.general_info.num_sensors}
                </div>
              </div>
            </div>
          </div>

          {/* Data Consistency Checks */}
          <div className="bg-gradient-to-r from-gray-800 to-gray-700 rounded-xl p-6 border border-gray-600 shadow-xl">
            <div className="flex items-center gap-3 mb-6">
              <div className="p-2 bg-indigo-600 rounded-lg">
                <CheckCircle className="text-white" size={24} />
              </div>
              <div>
                <h2 className="text-2xl font-semibold text-white">Data Consistency Checks</h2>
                <p className="text-gray-300">Duplicate detection and timestamp validation</p>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="bg-gray-900 rounded-lg p-4">
                <div className="flex items-center gap-2 mb-2">
                  {analyticsData.consistency_check.has_duplicates ? (
                    <XCircle className="text-red-400" size={20} />
                  ) : (
                    <CheckCircle className="text-green-400" size={20} />
                  )}
                  <span className="text-white font-medium">Duplicate Records</span>
                </div>
                {analyticsData.consistency_check.has_duplicates ? (
                  <div>
                    <p className="text-red-400">
                      {analyticsData.consistency_check.duplicate_count} duplicates found
                    </p>
                    <p className="text-gray-400 text-sm">
                      ({analyticsData.consistency_check.duplicate_percentage.toFixed(2)}% of total)
                    </p>
                  </div>
                ) : (
                  <p className="text-green-400">No duplicate records found</p>
                )}
              </div>

              <div className="bg-gray-900 rounded-lg p-4">
                <div className="flex items-center gap-2 mb-2">
                  {analyticsData.consistency_check.timestamps_consistent ? (
                    <CheckCircle className="text-green-400" size={20} />
                  ) : (
                    <XCircle className="text-red-400" size={20} />
                  )}
                  <span className="text-white font-medium">Timestamp Consistency</span>
                </div>
                <p className={analyticsData.consistency_check.timestamps_consistent ? "text-green-400" : "text-red-400"}>
                  {analyticsData.consistency_check.timestamps_consistent
                    ? "All timestamps are in order"
                    : "Timestamp inconsistency found"}
                </p>
              </div>
            </div>
          </div>

          {/* Data Completeness */}
          <div className="bg-gradient-to-r from-gray-800 to-gray-700 rounded-xl p-6 border border-gray-600 shadow-xl">
            <div className="flex items-center gap-3 mb-6">
              <div className="p-2 bg-purple-600 rounded-lg">
                <BarChart3 className="text-white" size={24} />
              </div>
              <div>
                <h2 className="text-2xl font-semibold text-white">Data Completeness</h2>
                <p className="text-gray-300">Completeness metrics across all sensors</p>
              </div>
            </div>

            <div className="mb-4">
              <div className="flex items-center justify-between mb-2">
                <span className="text-gray-300">Overall Data Completeness</span>
                <span className="text-2xl font-bold text-green-400">
                  {analyticsData.completeness_check.overall_completeness.toFixed(2)}%
                </span>
              </div>
              <div className="w-full bg-gray-700 rounded-full h-4">
                <div
                  className="bg-gradient-to-r from-green-600 to-emerald-600 h-4 rounded-full transition-all"
                  style={{ width: `${analyticsData.completeness_check.overall_completeness}%` }}
                ></div>
              </div>
            </div>

            {analyticsData.completeness_check.incomplete_sensors.length > 0 && (
              <div className="mt-4">
                <h3 className="text-lg font-semibold text-white mb-2">
                  Sensors Below {analyticsData.completeness_check.completeness_threshold}% Threshold
                </h3>
                <div className="overflow-auto max-h-48">
                  <table className="w-full text-sm">
                    <thead className="bg-gray-900 sticky top-0">
                      <tr>
                        <th className="text-left p-2 text-gray-300">Sensor</th>
                        <th className="text-right p-2 text-gray-300">Completeness</th>
                      </tr>
                    </thead>
                    <tbody>
                      {analyticsData.completeness_check.incomplete_sensors.map((sensor, idx) => (
                        <tr key={idx} className="border-t border-gray-700">
                          <td className="p-2 text-white">{sensor.sensor_name}</td>
                          <td className="p-2 text-right text-orange-400">
                            {sensor.completeness.toFixed(2)}%
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>

          {/* Outliers */}
          {analyticsData.outliers.filter(o => o.outlier_percentage > 0).length > 0 && (
            <div className="bg-gradient-to-r from-gray-800 to-gray-700 rounded-xl p-6 border border-gray-600 shadow-xl">
              <div className="flex items-center gap-3 mb-6">
                <div className="p-2 bg-yellow-600 rounded-lg">
                  <AlertTriangle className="text-white" size={24} />
                </div>
                <div>
                  <h2 className="text-2xl font-semibold text-white">Outliers Detection</h2>
                  <p className="text-gray-300">Sensors with outlier values (IQR method)</p>
                </div>
              </div>

              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={getOutliersChartData()}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis 
                    dataKey="sensor" 
                    stroke="#9CA3AF" 
                    angle={-45}
                    textAnchor="end"
                    height={100}
                  />
                  <YAxis stroke="#9CA3AF" />
                  <Tooltip 
                    contentStyle={{ backgroundColor: '#1F2937', border: '1px solid #374151' }}
                    formatter={(value: number) => [`${value}%`, 'Outliers']}
                  />
                  <Legend />
                  <Bar dataKey="percentage" fill="#F59E0B" name="Outlier %" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Accuracy Issues */}
          {analyticsData.accuracy_issues.length > 0 && (
            <div className="bg-gradient-to-r from-gray-800 to-gray-700 rounded-xl p-6 border border-gray-600 shadow-xl">
              <div className="flex items-center gap-3 mb-6">
                <div className="p-2 bg-red-600 rounded-lg">
                  <AlertCircle className="text-white" size={24} />
                </div>
                <div>
                  <h2 className="text-2xl font-semibold text-white">Data Accuracy Issues</h2>
                  <p className="text-gray-300">Sensors with threshold violations</p>
                </div>
              </div>

              <div className="overflow-auto max-h-96">
                <table className="w-full text-sm">
                  <thead className="bg-gray-900 sticky top-0">
                    <tr>
                      <th className="text-left p-2 text-gray-300">Sensor</th>
                      <th className="text-right p-2 text-gray-300">Issues %</th>
                      <th className="text-center p-2 text-gray-300">Type</th>
                      <th className="text-right p-2 text-gray-300">Low Threshold</th>
                      <th className="text-right p-2 text-gray-300">High Threshold</th>
                    </tr>
                  </thead>
                  <tbody>
                    {analyticsData.accuracy_issues.map((issue, idx) => (
                      <tr key={idx} className="border-t border-gray-700 hover:bg-gray-800">
                        <td className="p-2 text-white">{issue.sensor_name}</td>
                        <td className="p-2 text-right text-red-400 font-semibold">
                          {issue.issues_percentage.toFixed(2)}%
                        </td>
                        <td className="p-2 text-center text-gray-300">
                          {issue.threshold_type || '-'}
                        </td>
                        <td className="p-2 text-right text-gray-300">
                          {issue.low_threshold !== null ? issue.low_threshold.toFixed(2) : '-'}
                        </td>
                        <td className="p-2 text-right text-gray-300">
                          {issue.high_threshold !== null ? issue.high_threshold.toFixed(2) : '-'}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* Strong Correlations */}
          {analyticsData.strong_correlations.length > 0 && (
            <div className="bg-gradient-to-r from-gray-800 to-gray-700 rounded-xl p-6 border border-gray-600 shadow-xl">
              <div className="flex items-center gap-3 mb-6">
                <div className="p-2 bg-pink-600 rounded-lg">
                  <Activity className="text-white" size={24} />
                </div>
                <div>
                  <h2 className="text-2xl font-semibold text-white">Strong Correlations</h2>
                  <p className="text-gray-300">Sensor pairs with correlation &gt; 0.7</p>
                </div>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-3">
                {analyticsData.strong_correlations.map((corr, idx) => (
                  <div key={idx} className="bg-gray-900 rounded-lg p-3 border border-gray-700">
                    <div className="flex justify-between items-center">
                      <div className="text-sm">
                        <span className="text-white font-medium">{corr.sensor_a}</span>
                        <span className="text-gray-400 mx-2">↔</span>
                        <span className="text-white font-medium">{corr.sensor_b}</span>
                      </div>
                      <span className={`text-lg font-bold ${corr.correlation > 0 ? 'text-green-400' : 'text-red-400'}`}>
                        {corr.correlation.toFixed(2)}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}
        </>
      )}

      {/* Loading State */}
      {loading && (
        <div className="text-center py-12">
          <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-green-500 mx-auto"></div>
          <p className="text-gray-400 mt-4">Loading data...</p>
        </div>
      )}

      {/* No Data State */}
      {!loading && !analyticsData && selectedTable && (
        <div className="text-center py-12 bg-gray-800 rounded-xl border border-gray-700">
          <Shield className="mx-auto text-green-400 mb-4" size={64} />
          <h2 className="text-2xl font-bold text-white mb-2">No Analytics Data</h2>
          <p className="text-gray-400">Click "Analyze Quality" to generate data quality assessment</p>
        </div>
      )}
      
      <Footer />
    </div>
  )
}