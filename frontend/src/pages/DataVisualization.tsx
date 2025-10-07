import { useState, useEffect } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar, AreaChart, Area, ScatterChart, Scatter, Cell } from 'recharts'
import { ChevronDown, ChevronRight, TrendingUp, BarChart3, Activity, Settings, Database, FileText, Filter, Download, Zap, AlertCircle, CheckCircle, Target, Calendar, Layers } from 'lucide-react'
import SensorSelector from '../components/SensorSelector'

// Analytics API interfaces
interface SummaryStatistics {
  count: number
  mean: number
  std: number
  min: number
  q25: number
  q50: number
  q75: number
  max: number
}

interface CorrelationMatrix {
  columns: string[]
  data: number[][]
}

interface TimeSeriesPoint {
  timestamp: string
  original?: number
  rolling_mean?: number
  rolling_std?: number
}

interface HistogramBin {
  bin_start: number
  bin_end: number
  count: number
  density: number
}

interface BoxPlotStats {
  min: number
  q1: number
  median: number
  q3: number
  max: number
  outliers: number[]
}

interface SeasonalDecomposition {
  timestamps: string[]
  observed: number[]
  trend: (number | null)[]
  seasonal: (number | null)[]
  residual: (number | null)[]
}

interface AnomalyPoint {
  timestamp: string
  value: number
  z_score: number
}

interface SensorAnalysis {
  sensor_name: string
  summary_stats: SummaryStatistics
  time_series: TimeSeriesPoint[]
  histogram: HistogramBin[]
  box_plot: BoxPlotStats
  seasonal_decomposition?: SeasonalDecomposition
  anomalies: AnomalyPoint[]
}

interface VisualizationAnalytics {
  table_name: string
  selected_columns: string[]
  correlation_matrix: CorrelationMatrix
  sensor_analyses: SensorAnalysis[]
  processing_info: Record<string, any>
}

// Legacy interfaces for backward compatibility
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

export default function DataVisualization() {
  const [tables, setTables] = useState<string[]>([])
  const [selectedTable, setSelectedTable] = useState<string>('')
  const [preprocessedData, setPreprocessedData] = useState<PreprocessedData | null>(null)
  const [tagsData, setTagsData] = useState<TagsData | null>(null)
  const [selectedColumns, setSelectedColumns] = useState<string[]>([])
  const [loading, setLoading] = useState<boolean>(false)
  const [error, setError] = useState<string>('')
  
  // New analytics state
  const [analyticsData, setAnalyticsData] = useState<VisualizationAnalytics | null>(null)
  const [analyticsLoading, setAnalyticsLoading] = useState<boolean>(false)
  const [activeAnalysisType, setActiveAnalysisType] = useState<'summary' | 'correlation' | 'timeseries' | 'histogram' | 'boxplot' | 'seasonal' | 'anomalies'>('summary')
  
  // Date range filtering
  const [dateFrom, setDateFrom] = useState<string>('')
  const [dateTo, setDateTo] = useState<string>('')
  const [availableDateRange, setAvailableDateRange] = useState<{min: string, max: string} | null>(null)
  const dataSource = 'aggregated' // Always use aggregated data

  const API_BASE = 'http://localhost:8000'

  // Fetch tables on component mount
  useEffect(() => {
    fetchTables()
  }, [])

  // Load data when table is selected
  useEffect(() => {
    if (selectedTable) {
      // Clear date filters when switching tables so new defaults can be set
      setDateFrom('')
      setDateTo('')
      loadData()
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
      const allTables = await response.json()
      // Filter tables with "hours" in name (like Streamlit version)
      const filteredTables = allTables.filter((table: string) => 
        table.toLowerCase().includes('hours')
      )
      setTables(filteredTables)
    } catch (err) {
      setError('Error fetching tables: ' + (err as Error).message)
    }
  }

  const loadData = async () => {
    if (!selectedTable) return

    setLoading(true)
    setError('')
    
    try {
      // Fetch preprocessed data
      const preprocessedResponse = await fetch(`${API_BASE}/data/preprocessed?table=${selectedTable}&limit=5000`)
      if (!preprocessedResponse.ok) throw new Error('Failed to fetch preprocessed data')
      const preprocessedResult = await preprocessedResponse.json()
      
      // Debug: Log the data structure
      console.log('Preprocessed data:', preprocessedResult)
      console.log('Rows count:', preprocessedResult?.readings?.rows?.length)
      console.log('Columns:', preprocessedResult?.readings?.columns)
      
      if (!preprocessedResult?.readings?.rows || preprocessedResult.readings.rows.length === 0) {
        setError('No data available for the selected table')
      }
      
      setPreprocessedData(preprocessedResult)

      // Fetch tags data filtered by selected table
      const tagsResponse = await fetch(`${API_BASE}/tags?table=${selectedTable}`)
      if (!tagsResponse.ok) throw new Error('Failed to fetch tags data')
      const tagsResult = await tagsResponse.json()
      setTagsData(tagsResult)

    } catch (err) {
      setError('Error loading data: ' + (err as Error).message)
    } finally {
      setLoading(false)
    }
  }

  const fetchAnalytics = async () => {
    if (!selectedTable || selectedColumns.length === 0) return

    setAnalyticsLoading(true)
    setError('')
    
    try {
      const params = new URLSearchParams({
        table: selectedTable,
        limit: '5000',
        rolling_window: '24',
        anomaly_threshold: '2.0',
        seasonal_period: '24',
        data_source: dataSource
      })
      
      // Add date range if specified
      if (dateFrom) {
        params.append('date_from', dateFrom)
      }
      if (dateTo) {
        params.append('date_to', dateTo)
      }
      
      // Add each column as a separate parameter
      selectedColumns.forEach(col => {
        params.append('columns', col)
      })

      const response = await fetch(`${API_BASE}/analytics/visualization?${params.toString()}`)
      if (!response.ok) throw new Error('Failed to fetch analytics data')
      
      const analyticsResult: VisualizationAnalytics = await response.json()
      setAnalyticsData(analyticsResult)

    } catch (err) {
      setError('Error loading analytics: ' + (err as Error).message)
    } finally {
      setAnalyticsLoading(false)
    }
  }

  const getAvailableSensors = () => {
    if (!preprocessedData) return []
    
    const numericColumns = preprocessedData.readings.columns.filter(col => {
      if (col === 'timestamp' || col.toLowerCase() === 'timestamp') return false
      
      // Check multiple rows to find numeric values (not just the first row)
      for (let i = 0; i < Math.min(10, preprocessedData.readings.rows.length); i++) {
        const sampleValue = preprocessedData.readings.rows[i]?.[col]
        if (sampleValue !== null && sampleValue !== undefined && typeof sampleValue === 'number') {
          return true
        }
      }
      return false
    })

    return numericColumns
  }

  const getSensorInfo = (sensor: string) => {
    if (!tagsData) return null

    // Extract tag name from column (remove prefixes like "col", "sum_", etc.)
    let tagName = sensor
    if (sensor.startsWith('sum_col') || sensor.startsWith('count_col') || sensor.startsWith('min_col') || sensor.startsWith('max_col')) {
      tagName = sensor.split('_', 2)[1] // Remove "sum_", "count_", etc.
      if (tagName && tagName.startsWith('col')) {
        tagName = tagName.substring(3) // Remove "col" prefix
      }
    } else if (sensor.startsWith('col')) {
      tagName = sensor.substring(3) // Remove "col" prefix
    }

    const tag = tagsData.rows.find(row =>
      row.tag && row.tag.toLowerCase() === tagName.toLowerCase()
    )

    if (!tag) return null

    return {
      name: sensor,
      category: tag.category || undefined,
      description: tag.description || undefined,
      unit: tag.engineering_units || tag.unit || undefined
    }
  }

  const renderAnalyticsContent = () => {
    if (!analyticsData) return null

    switch (activeAnalysisType) {
      case 'summary':
        return renderSummaryStatistics()
      case 'correlation':
        return renderCorrelationMatrix()
      case 'timeseries':
        return renderTimeSeriesAnalysis()
      case 'histogram':
        return renderHistogramAnalysis()
      case 'boxplot':
        return renderBoxPlotAnalysis()
      case 'seasonal':
        return renderSeasonalDecomposition()
      case 'anomalies':
        return renderAnomalyDetection()
      default:
        return null
    }
  }

  const renderSummaryStatistics = () => {
    if (!analyticsData) return null

    return (
      <div className="space-y-6">
        <h3 className="text-xl font-semibold text-white mb-4">Summary Statistics</h3>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {analyticsData.sensor_analyses.map((analysis) => (
            <div key={analysis.sensor_name} className="bg-gray-800/80 rounded-lg p-6 border border-gray-600">
              <h4 className="text-lg font-medium text-white mb-4">{analysis.sensor_name}</h4>
              <div className="grid grid-cols-2 gap-4 text-sm">
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Count:</span>
                    <span className="text-white">{analysis.summary_stats.count.toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Mean:</span>
                    <span className="text-white">{analysis.summary_stats.mean.toFixed(3)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Std Dev:</span>
                    <span className="text-white">{analysis.summary_stats.std.toFixed(3)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Min:</span>
                    <span className="text-white">{analysis.summary_stats.min.toFixed(3)}</span>
                  </div>
                </div>
                <div className="space-y-2">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Q1:</span>
                    <span className="text-white">{analysis.summary_stats.q25.toFixed(3)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Median:</span>
                    <span className="text-white">{analysis.summary_stats.q50.toFixed(3)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Q3:</span>
                    <span className="text-white">{analysis.summary_stats.q75.toFixed(3)}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Max:</span>
                    <span className="text-white">{analysis.summary_stats.max.toFixed(3)}</span>
                  </div>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    )
  }

  const renderCorrelationMatrix = () => {
    if (!analyticsData) return null

    const { correlation_matrix } = analyticsData
    const size = correlation_matrix.columns.length

    return (
      <div className="space-y-6">
        <h3 className="text-xl font-semibold text-white mb-4">Correlation Matrix</h3>
        <div className="bg-gray-800/80 rounded-lg p-6 border border-gray-600">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr>
                  <th className="text-left p-2 text-gray-300"></th>
                  {correlation_matrix.columns.map(col => (
                    <th key={col} className="text-center p-2 text-gray-300 font-medium min-w-[80px]">
                      {col}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {correlation_matrix.data.map((row, i) => (
                  <tr key={i}>
                    <td className="text-left p-2 text-gray-300 font-medium">
                      {correlation_matrix.columns[i]}
                    </td>
                    {row.map((value, j) => (
                      <td key={j} className="text-center p-2">
                        <div 
                          className={`
                            px-2 py-1 rounded text-xs font-medium
                            ${Math.abs(value) > 0.7 
                              ? 'bg-red-600 text-white' 
                              : Math.abs(value) > 0.5 
                                ? 'bg-yellow-600 text-white'
                                : 'bg-gray-600 text-gray-200'
                            }
                          `}
                        >
                          {value.toFixed(2)}
                        </div>
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="mt-4 text-xs text-gray-400">
            <p><span className="bg-red-600 px-2 py-1 rounded mr-2"></span>Strong correlation (|r| &gt; 0.7)</p>
            <p><span className="bg-yellow-600 px-2 py-1 rounded mr-2"></span>Moderate correlation (|r| &gt; 0.5)</p>
          </div>
        </div>
      </div>
    )
  }

  const renderTimeSeriesAnalysis = () => {
    if (!analyticsData) return null

    return (
      <div className="space-y-6">
        <h3 className="text-xl font-semibold text-white mb-4">Time Series Analysis</h3>
        {analyticsData.sensor_analyses.map((analysis) => (
          <div key={analysis.sensor_name} className="bg-gray-800/80 rounded-lg p-6 border border-gray-600">
            <h4 className="text-lg font-medium text-white mb-4">{analysis.sensor_name}</h4>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={analysis.time_series}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis 
                    dataKey="timestamp" 
                    stroke="#9CA3AF"
                    fontSize={12}
                    tick={{ fill: '#9CA3AF' }}
                  />
                  <YAxis 
                    stroke="#9CA3AF"
                    fontSize={12}
                    tick={{ fill: '#9CA3AF' }}
                  />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: '#1F2937', 
                      border: '1px solid #374151',
                      borderRadius: '8px',
                      color: '#F3F4F6'
                    }}
                  />
                  <Legend />
                  <Line 
                    type="monotone" 
                    dataKey="original" 
                    stroke="#3B82F6"
                    strokeWidth={1}
                    dot={false}
                    name="Original"
                  />
                  <Line 
                    type="monotone" 
                    dataKey="rolling_mean" 
                    stroke="#EF4444"
                    strokeWidth={2}
                    dot={false}
                    name="Rolling Mean"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        ))}
      </div>
    )
  }

  const renderHistogramAnalysis = () => {
    if (!analyticsData) return null

    return (
      <div className="space-y-6">
        <h3 className="text-xl font-semibold text-white mb-4">Distribution Analysis</h3>
        {analyticsData.sensor_analyses.map((analysis) => (
          <div key={analysis.sensor_name} className="bg-gray-800/80 rounded-lg p-6 border border-gray-600">
            <h4 className="text-lg font-medium text-white mb-4">{analysis.sensor_name}</h4>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={analysis.histogram}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                  <XAxis 
                    dataKey="bin_start" 
                    stroke="#9CA3AF"
                    fontSize={12}
                    tick={{ fill: '#9CA3AF' }}
                  />
                  <YAxis 
                    stroke="#9CA3AF"
                    fontSize={12}
                    tick={{ fill: '#9CA3AF' }}
                  />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: '#1F2937', 
                      border: '1px solid #374151',
                      borderRadius: '8px',
                      color: '#F3F4F6'
                    }}
                  />
                  <Bar 
                    dataKey="count" 
                    fill="#10B981"
                    name="Frequency"
                  />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        ))}
      </div>
    )
  }

  const renderBoxPlotAnalysis = () => {
    if (!analyticsData) return null

    return (
      <div className="space-y-6">
        <h3 className="text-xl font-semibold text-white mb-4">Box Plot Analysis</h3>
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {analyticsData.sensor_analyses.map((analysis) => (
            <div key={analysis.sensor_name} className="bg-gray-800/80 rounded-lg p-6 border border-gray-600">
              <h4 className="text-lg font-medium text-white mb-4">{analysis.sensor_name}</h4>
              <div className="space-y-3">
                <div className="flex justify-between text-sm">
                  <span className="text-gray-400">Min:</span>
                  <span className="text-white">{analysis.box_plot.min.toFixed(3)}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-400">Q1:</span>
                  <span className="text-white">{analysis.box_plot.q1.toFixed(3)}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-400">Median:</span>
                  <span className="text-white font-medium">{analysis.box_plot.median.toFixed(3)}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-400">Q3:</span>
                  <span className="text-white">{analysis.box_plot.q3.toFixed(3)}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-400">Max:</span>
                  <span className="text-white">{analysis.box_plot.max.toFixed(3)}</span>
                </div>
                <div className="flex justify-between text-sm">
                  <span className="text-gray-400">Outliers:</span>
                  <span className="text-orange-400">{analysis.box_plot.outliers.length}</span>
                </div>
              </div>
            </div>
          ))}
        </div>
      </div>
    )
  }

  const renderSeasonalDecomposition = () => {
    if (!analyticsData) return null

    return (
      <div className="space-y-6">
        <h3 className="text-xl font-semibold text-white mb-4">Seasonal Decomposition</h3>
        {analyticsData.sensor_analyses.map((analysis) => {
          if (!analysis.seasonal_decomposition) {
            return (
              <div key={analysis.sensor_name} className="bg-gray-800/80 rounded-lg p-6 border border-gray-600">
                <h4 className="text-lg font-medium text-white mb-4">{analysis.sensor_name}</h4>
                <p className="text-gray-400">Not enough data for seasonal decomposition</p>
              </div>
            )
          }

          const decomp = analysis.seasonal_decomposition
          const chartData = decomp.timestamps.map((timestamp, i) => ({
            timestamp,
            observed: decomp.observed[i],
            trend: decomp.trend[i],
            seasonal: decomp.seasonal[i],
            residual: decomp.residual[i]
          }))

          return (
            <div key={analysis.sensor_name} className="bg-gray-800/80 rounded-lg p-6 border border-gray-600">
              <h4 className="text-lg font-medium text-white mb-4">{analysis.sensor_name}</h4>
              <div className="space-y-6">
                {['observed', 'trend', 'seasonal', 'residual'].map((component) => (
                  <div key={component} className="h-48">
                    <h5 className="text-sm font-medium text-gray-300 mb-2 capitalize">{component}</h5>
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={chartData}>
                        <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                        <XAxis 
                          dataKey="timestamp" 
                          stroke="#9CA3AF"
                          fontSize={10}
                          tick={{ fill: '#9CA3AF' }}
                        />
                        <YAxis 
                          stroke="#9CA3AF"
                          fontSize={10}
                          tick={{ fill: '#9CA3AF' }}
                        />
                        <Tooltip 
                          contentStyle={{ 
                            backgroundColor: '#1F2937', 
                            border: '1px solid #374151',
                            borderRadius: '8px',
                            color: '#F3F4F6'
                          }}
                        />
                        <Line 
                          type="monotone" 
                          dataKey={component} 
                          stroke={component === 'observed' ? '#3B82F6' : component === 'trend' ? '#EF4444' : component === 'seasonal' ? '#10B981' : '#F59E0B'}
                          strokeWidth={1}
                          dot={false}
                        />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>
                ))}
              </div>
            </div>
          )
        })}
      </div>
    )
  }

  const renderAnomalyDetection = () => {
    if (!analyticsData) return null

    return (
      <div className="space-y-6">
        <h3 className="text-xl font-semibold text-white mb-4">Anomaly Detection</h3>
        {analyticsData.sensor_analyses.map((analysis) => (
          <div key={analysis.sensor_name} className="bg-gray-800/80 rounded-lg p-6 border border-gray-600">
            <h4 className="text-lg font-medium text-white mb-4">{analysis.sensor_name}</h4>
            
            <div className="mb-4 flex items-center gap-4">
              <div className="bg-red-900/30 border border-red-600 rounded-lg px-4 py-2">
                <span className="text-red-300 font-medium">{analysis.anomalies.length} anomalies detected</span>
              </div>
            </div>

            {analysis.anomalies.length > 0 && (
              <div className="space-y-4">
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <LineChart 
                      data={(() => {
                        // Create combined dataset with anomaly markers
                        return analysis.time_series.map(point => {
                          const anomaly = analysis.anomalies.find(a => a.timestamp === point.timestamp)
                          return {
                            ...point,
                            anomaly_value: anomaly ? anomaly.value : null
                          }
                        })
                      })()}
                    >
                      <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                      <XAxis 
                        dataKey="timestamp" 
                        stroke="#9CA3AF"
                        fontSize={12}
                        tick={{ fill: '#9CA3AF' }}
                      />
                      <YAxis 
                        stroke="#9CA3AF"
                        fontSize={12}
                        tick={{ fill: '#9CA3AF' }}
                      />
                      <Tooltip 
                        contentStyle={{ 
                          backgroundColor: '#1F2937', 
                          border: '1px solid #374151',
                          borderRadius: '8px',
                          color: '#F3F4F6'
                        }}
                      />
                      <Legend />
                      {/* Main time series line */}
                      <Line 
                        type="monotone" 
                        dataKey="original" 
                        stroke="#3B82F6"
                        strokeWidth={2}
                        dot={false}
                        name="Sensor Values"
                      />
                      {/* Anomaly points overlaid */}
                      <Line 
                        type="monotone" 
                        dataKey="anomaly_value" 
                        stroke="none"
                        strokeWidth={0}
                        dot={{ fill: '#EF4444', strokeWidth: 2, r: 4 }}
                        name="Anomalies"
                        connectNulls={false}
                      />
                    </LineChart>
                  </ResponsiveContainer>
                </div>

                <div className="max-h-48 overflow-y-auto">
                  <table className="w-full text-sm">
                    <thead className="bg-gray-700 sticky top-0">
                      <tr>
                        <th className="text-left p-2 text-gray-300">Timestamp</th>
                        <th className="text-left p-2 text-gray-300">Value</th>
                        <th className="text-left p-2 text-gray-300">Z-Score</th>
                      </tr>
                    </thead>
                    <tbody>
                      {analysis.anomalies.slice(0, 20).map((anomaly, i) => (
                        <tr key={i} className="border-b border-gray-700">
                          <td className="p-2 text-gray-200">{new Date(anomaly.timestamp).toLocaleString()}</td>
                          <td className="p-2 text-white">{anomaly.value.toFixed(3)}</td>
                          <td className="p-2 text-red-400">{anomaly.z_score.toFixed(2)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        ))}
      </div>
    )
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-center gap-3">
        <div className="p-3 bg-gradient-to-r from-blue-600 to-indigo-600 rounded-lg shadow-lg">
          <TrendingUp className="text-white" size={32} />
        </div>
        <div>
          <h1 className="text-4xl font-bold text-white">Data Visualization</h1>
          <p className="text-gray-400 mt-1">Visualize your sensor data with interactive charts and explore various data patterns</p>
        </div>
      </div>

      {error && (
        <div className="bg-red-900/50 border border-red-700 rounded-xl p-4 flex items-center gap-3 animate-in slide-in-from-top duration-300">
          <AlertCircle className="text-red-400 flex-shrink-0" size={20} />
          <span className="text-red-200">{error}</span>
        </div>
      )}

      {/* Configuration Panel */}
      <div className="bg-gradient-to-r from-gray-800 to-gray-700 rounded-xl p-6 border border-gray-600 shadow-xl">
        <div className="flex items-center gap-3 mb-6">
          <div className="p-2 bg-blue-600 rounded-lg">
            <Settings className="text-white" size={24} />
          </div>
          <div>
            <h2 className="text-2xl font-semibold text-white">Analysis Configuration</h2>
            <p className="text-gray-300">Configure table selection and sensor filters</p>
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
              className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-white focus:ring-2 focus:ring-blue-500 focus:border-transparent"
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
              onClick={fetchAnalytics}
              disabled={!selectedTable || selectedColumns.length === 0 || analyticsLoading}
              className="w-full bg-gradient-to-r from-blue-600 to-indigo-600 hover:from-blue-700 hover:to-indigo-700 disabled:from-gray-600 disabled:to-gray-700 text-white font-semibold py-2 px-4 rounded-lg transition-all duration-200 flex items-center justify-center"
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

        {loading && (
          <div className="mt-4 flex items-center gap-3 text-blue-400">
            <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-blue-400"></div>
            <span>Loading data...</span>
          </div>
        )}

        {/* Column Selection */}
        {preprocessedData && (
          <SensorSelector
            sensors={getAvailableSensors()}
            selectedSensors={selectedColumns}
            onSelectionChange={setSelectedColumns}
            getSensorInfo={getSensorInfo}
            accentColor="blue"
          />
        )}
      </div>

      {/* Data Filtering Options */}
      {selectedTable && (
        <div className="bg-gradient-to-r from-gray-800 to-gray-700 rounded-xl p-6 border border-gray-600 shadow-xl">
          <div className="flex items-center gap-3 mb-6">
            <div className="p-2 bg-orange-600 rounded-lg">
              <Filter className="text-white" size={24} />
            </div>
            <div>
              <h2 className="text-2xl font-semibold text-white">Data Filtering</h2>
              <p className="text-gray-300">Filter your data by date range</p>
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
                Filter data from this date
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
                Filter data until this date
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

      {/* Advanced Analytics Section */}
      {analyticsData && (
        <div className="bg-gradient-to-r from-gray-800 to-gray-700 rounded-xl p-6 border border-gray-600 shadow-xl">
          <div className="flex items-center gap-3 mb-6">
            <div className="p-2 bg-indigo-600 rounded-lg">
              <Target className="text-white" size={24} />
            </div>
            <div>
              <h2 className="text-2xl font-semibold text-white">Advanced Analytics</h2>
              <p className="text-gray-300">Statistical analysis and insights for your sensor data</p>
            </div>
          </div>

          {/* Analytics Type Selection */}
          <div className="flex flex-wrap gap-2 mb-6">
            {[
              { type: 'summary' as const, label: 'Summary Stats', icon: Database },
              { type: 'correlation' as const, label: 'Correlation', icon: Layers },
              { type: 'timeseries' as const, label: 'Time Series', icon: TrendingUp },
              { type: 'histogram' as const, label: 'Distribution', icon: BarChart3 },
              { type: 'boxplot' as const, label: 'Box Plot', icon: Activity },
              { type: 'seasonal' as const, label: 'Seasonal', icon: Activity },
              { type: 'anomalies' as const, label: 'Anomalies', icon: AlertCircle },
            ].map(({ type, label, icon: Icon }) => (
              <button
                key={type}
                onClick={() => setActiveAnalysisType(type)}
                className={`
                  flex items-center gap-2 px-4 py-2 rounded-lg transition-all
                  ${activeAnalysisType === type 
                    ? 'bg-indigo-600 text-white' 
                    : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                  }
                `}
              >
                <Icon size={16} />
                {label}
              </button>
            ))}
          </div>

          {/* Analytics Loading State */}
          {analyticsLoading && (
            <div className="flex items-center justify-center py-12">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-400"></div>
              <span className="ml-3 text-indigo-400">Processing analytics...</span>
            </div>
          )}

          {/* Analytics Content */}
          {!analyticsLoading && (
            <div className="bg-gray-900/50 rounded-lg border border-gray-600 p-6">
              {renderAnalyticsContent()}
            </div>
          )}
        </div>
      )}

      {/* Loading State */}
      {loading && (
        <div className="text-center py-12">
          <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-blue-500 mx-auto"></div>
          <p className="text-gray-400 mt-4">Loading data...</p>
        </div>
      )}

      {/* No Data State */}
      {!loading && !analyticsData && selectedTable && (
        <div className="text-center py-12 bg-gray-800 rounded-xl border border-gray-700">
          <Target className="mx-auto text-blue-400 mb-4" size={64} />
          <h2 className="text-2xl font-bold text-white mb-2">No Analytics Data</h2>
          <p className="text-gray-400">Select sensors and click "Analyze" to generate advanced analytics</p>
        </div>
      )}
    </div>
  )
}