import { useState, useEffect } from 'react'
import { Database, AlertTriangle, TrendingUp, BarChart3, Activity, Clock } from 'lucide-react'
import {
  LineChart,
  Line,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine
} from 'recharts'
import SensorSelector from '../components/SensorSelector'
import Footer from '../components/Footer'

interface MissingInterval {
  start: string
  end: string
  duration_hours: number
}

interface SensorMissingStats {
  sensor_name: string
  expected_readings: number
  actual_readings: number
  missing_readings: number
  missing_percentage: number
  missing_intervals: MissingInterval[]
}

interface MissingValuesAnalytics {
  table_name: string
  selected_columns: string[]
  total_expected_readings: number
  total_actual_readings: number
  total_missing_readings: number
  overall_missing_percentage: number
  sensor_stats: SensorMissingStats[]
  processing_info: {
    rows_analyzed: number
    sensors_analyzed: number
    original_freq_sec: number
    expected_readings_per_hour: number
    time_range_hours: number
  }
}

interface TagsData {
  columns: string[]
  rows: Record<string, any>[]
}

export default function MissingValues() {
  const [tables, setTables] = useState<string[]>([])
  const [selectedTable, setSelectedTable] = useState<string>('')
  const [preprocessedData, setPreprocessedData] = useState<any>(null)
  const [tagsData, setTagsData] = useState<TagsData | null>(null)
  const [selectedColumns, setSelectedColumns] = useState<string[]>([])
  const [analyticsData, setAnalyticsData] = useState<MissingValuesAnalytics | null>(null)
  const [loading, setLoading] = useState<boolean>(false)
  const [analyticsLoading, setAnalyticsLoading] = useState<boolean>(false)

  const API_BASE = 'http://localhost:8000'

  useEffect(() => {
    fetchTables()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    if (selectedTable) {
      loadData()
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selectedTable])

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
    } catch (error) {
      console.error('Error fetching tables:', error)
    }
  }

  const loadData = async () => {
    if (!selectedTable) return
    
    setLoading(true)
    try {
      // Fetch raw data to get available COUNT columns
      const response = await fetch(`${API_BASE}/data?table=${selectedTable}&limit=10`)
      const data = await response.json()
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
    } catch (error) {
      console.error('Error loading data:', error)
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

  const fetchMissingValuesAnalytics = async () => {
    if (!selectedTable) return

    setAnalyticsLoading(true)
    try {
      const requestBody = {
        table: selectedTable,
        columns: selectedColumns.length > 0 ? selectedColumns : null
      }

      const response = await fetch(`${API_BASE}/analytics/missing-values`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify(requestBody)
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      setAnalyticsData(data)
    } catch (error) {
      console.error('Error fetching missing values analytics:', error)
      alert('Failed to fetch analytics. Please try again.')
    } finally {
      setAnalyticsLoading(false)
    }
  }

  const getAvailableSensors = (): string[] => {
    if (!preprocessedData || !preprocessedData.columns) return []
    return preprocessedData.columns
      .filter((col: string) => col.startsWith('COUNT_COL') && !col.includes('_ISVALID'))
      .map((col: string) => col.replace('COUNT_COL', '').replace('COUNT_', ''))
  }

  // Prepare chart data for missing percentages by sensor
  const getMissingPercentageChartData = () => {
    if (!analyticsData) return []
    
    return analyticsData.sensor_stats
      .map(stat => ({
        sensor: stat.sensor_name,
        missing_percentage: parseFloat(stat.missing_percentage.toFixed(2)),
        actual_readings: stat.actual_readings,
        missing_readings: stat.missing_readings
      }))
      .sort((a, b) => b.missing_percentage - a.missing_percentage)
      .slice(0, 20) // Show top 20
  }

  // Get top sensors with most missing data
  const getTopMissingSensors = () => {
    if (!analyticsData) return []
    
    return analyticsData.sensor_stats
      .filter(stat => stat.missing_readings > 0)
      .sort((a, b) => b.missing_readings - a.missing_readings)
      .slice(0, 10)
  }

  // Get sensors with longest missing intervals
  const getSensorsWithLongestIntervals = () => {
    if (!analyticsData) return []
    
    const sensorsWithIntervals = analyticsData.sensor_stats
      .filter(stat => stat.missing_intervals.length > 0)
      .map(stat => {
        const longestInterval = stat.missing_intervals.reduce((max, interval) => 
          interval.duration_hours > max.duration_hours ? interval : max
        , stat.missing_intervals[0])
        
        return {
          sensor: stat.sensor_name,
          longest_duration: longestInterval.duration_hours,
          interval_start: longestInterval.start,
          interval_end: longestInterval.end,
          total_intervals: stat.missing_intervals.length
        }
      })
      .sort((a, b) => b.longest_duration - a.longest_duration)
      .slice(0, 10)
    
    return sensorsWithIntervals
  }

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="flex items-center gap-3">
        <div className="p-3 bg-gradient-to-r from-orange-600 to-red-600 rounded-lg shadow-lg">
          <AlertTriangle className="text-white" size={32} />
        </div>
        <div>
          <h1 className="text-4xl font-bold text-white">Missing Values Analysis</h1>
          <p className="text-gray-400 mt-1">Analyze data completeness and identify missing reading periods</p>
        </div>
      </div>

      {/* Configuration Panel */}
      <div className="bg-gradient-to-r from-gray-800 to-gray-700 rounded-xl p-6 border border-gray-600 shadow-xl">
        <div className="flex items-center gap-3 mb-6">
          <div className="p-2 bg-orange-600 rounded-lg">
            <Database className="text-white" size={24} />
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
              onClick={fetchMissingValuesAnalytics}
              disabled={!selectedTable || analyticsLoading}
              className="w-full bg-gradient-to-r from-orange-600 to-red-600 hover:from-orange-700 hover:to-red-700 disabled:from-gray-600 disabled:to-gray-700 text-white font-semibold py-2 px-4 rounded-lg transition-all duration-200 flex items-center justify-center"
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
            sensors={getAvailableSensors()}
            selectedSensors={selectedColumns}
            onSelectionChange={setSelectedColumns}
            getSensorInfo={getSensorInfo}
            accentColor="orange"
          />
        )}
      </div>

      {/* Analytics Results */}
      {analyticsData && (
        <>
          {/* Overall Statistics */}
          <div className="bg-gradient-to-r from-gray-800 to-gray-700 rounded-xl p-6 border border-gray-600 shadow-xl">
            <div className="flex items-center gap-3 mb-6">
              <div className="p-2 bg-orange-600 rounded-lg">
                <BarChart3 className="text-white" size={24} />
              </div>
              <div>
                <h2 className="text-2xl font-semibold text-white">Overall Statistics</h2>
                <p className="text-gray-300">Summary of missing data across all sensors</p>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="bg-gray-900 rounded-lg p-4">
                <div className="text-gray-400 text-sm mb-1">Total Missing</div>
                <div className="text-2xl font-bold text-orange-400">
                  {analyticsData.total_missing_readings.toLocaleString()}
                </div>
                <div className="text-xs text-gray-500 mt-1">readings</div>
              </div>
              <div className="bg-gray-900 rounded-lg p-4">
                <div className="text-gray-400 text-sm mb-1">Missing Rate</div>
                <div className="text-2xl font-bold text-red-400">
                  {analyticsData.overall_missing_percentage.toFixed(2)}%
                </div>
                <div className="text-xs text-gray-500 mt-1">of expected data</div>
              </div>
              <div className="bg-gray-900 rounded-lg p-4">
                <div className="text-gray-400 text-sm mb-1">Sensors Analyzed</div>
                <div className="text-2xl font-bold text-blue-400">
                  {analyticsData.processing_info.sensors_analyzed}
                </div>
                <div className="text-xs text-gray-500 mt-1">sensors</div>
              </div>
              <div className="bg-gray-900 rounded-lg p-4">
                <div className="text-gray-400 text-sm mb-1">Time Range</div>
                <div className="text-2xl font-bold text-green-400">
                  {analyticsData.processing_info.time_range_hours.toFixed(1)}
                </div>
                <div className="text-xs text-gray-500 mt-1">hours</div>
              </div>
            </div>
          </div>

          {/* Missing Percentage by Sensor Chart */}
          <div className="bg-gradient-to-r from-gray-800 to-gray-700 rounded-xl p-6 border border-gray-600 shadow-xl">
            <div className="flex items-center gap-3 mb-6">
              <div className="p-2 bg-orange-600 rounded-lg">
                <Activity className="text-white" size={24} />
              </div>
              <div>
                <h2 className="text-2xl font-semibold text-white">Missing Data by Sensor</h2>
                <p className="text-gray-300">Top 20 sensors with highest missing percentage</p>
              </div>
            </div>

            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={getMissingPercentageChartData()}>
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
                  formatter={(value: any, name: string) => {
                    if (name === 'missing_percentage') return [`${value}%`, 'Missing %']
                    return [value, name]
                  }}
                />
                <Legend />
                <Bar dataKey="missing_percentage" fill="#F97316" name="Missing %" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Top Sensors with Most Missing Data */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Missing Readings Table */}
            <div className="bg-gradient-to-r from-gray-800 to-gray-700 rounded-xl p-6 border border-gray-600 shadow-xl">
              <div className="flex items-center gap-3 mb-4">
                <div className="p-2 bg-red-600 rounded-lg">
                  <AlertTriangle className="text-white" size={20} />
                </div>
                <div>
                  <h3 className="text-xl font-semibold text-white">Most Missing Readings</h3>
                  <p className="text-sm text-gray-400">Sensors with highest missing counts</p>
                </div>
              </div>
              
              <div className="overflow-auto max-h-96">
                <table className="w-full text-sm">
                  <thead className="bg-gray-900 sticky top-0">
                    <tr>
                      <th className="text-left p-2 text-gray-300">Sensor</th>
                      <th className="text-right p-2 text-gray-300">Missing</th>
                      <th className="text-right p-2 text-gray-300">%</th>
                    </tr>
                  </thead>
                  <tbody>
                    {getTopMissingSensors().map((sensor, idx) => (
                      <tr key={idx} className="border-t border-gray-700 hover:bg-gray-800">
                        <td className="p-2 text-white">{sensor.sensor_name}</td>
                        <td className="p-2 text-right text-orange-400">
                          {sensor.missing_readings.toLocaleString()}
                        </td>
                        <td className="p-2 text-right text-red-400">
                          {sensor.missing_percentage.toFixed(2)}%
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Longest Missing Intervals */}
            <div className="bg-gradient-to-r from-gray-800 to-gray-700 rounded-xl p-6 border border-gray-600 shadow-xl">
              <div className="flex items-center gap-3 mb-4">
                <div className="p-2 bg-orange-600 rounded-lg">
                  <Clock className="text-white" size={20} />
                </div>
                <div>
                  <h3 className="text-xl font-semibold text-white">Longest Missing Intervals</h3>
                  <p className="text-sm text-gray-400">Sensors with longest data gaps</p>
                </div>
              </div>
              
              <div className="overflow-auto max-h-96">
                <table className="w-full text-sm">
                  <thead className="bg-gray-900 sticky top-0">
                    <tr>
                      <th className="text-left p-2 text-gray-300">Sensor</th>
                      <th className="text-right p-2 text-gray-300">Duration</th>
                      <th className="text-right p-2 text-gray-300">Intervals</th>
                    </tr>
                  </thead>
                  <tbody>
                    {getSensorsWithLongestIntervals().map((sensor, idx) => (
                      <tr key={idx} className="border-t border-gray-700 hover:bg-gray-800">
                        <td className="p-2 text-white">{sensor.sensor}</td>
                        <td className="p-2 text-right text-orange-400">
                          {sensor.longest_duration.toFixed(1)}h
                        </td>
                        <td className="p-2 text-right text-gray-400">
                          {sensor.total_intervals}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          </div>
        </>
      )}

      {/* Loading State */}
      {loading && (
        <div className="text-center py-12">
          <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-orange-500 mx-auto"></div>
          <p className="text-gray-400 mt-4">Loading data...</p>
        </div>
      )}

      {/* No Data State */}
      {!loading && !analyticsData && selectedTable && (
        <div className="text-center py-12 bg-gray-800 rounded-xl border border-gray-700">
          <AlertTriangle className="mx-auto text-orange-400 mb-4" size={64} />
          <h2 className="text-2xl font-bold text-white mb-2">No Analytics Data</h2>
          <p className="text-gray-400">Click "Analyze" to generate missing values statistics</p>
        </div>
      )}
      
      <Footer />
    </div>
  )
}