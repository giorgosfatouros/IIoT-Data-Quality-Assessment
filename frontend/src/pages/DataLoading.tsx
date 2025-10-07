import { useState, useEffect } from 'react'
import { ChevronDown, ChevronRight, Download, Loader2, AlertCircle, CheckCircle, Database, Settings, FileText, Zap, TrendingUp, Upload, CloudUpload, Plus } from 'lucide-react'

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

interface AggregationFrequency {
  aggregation_frequency_seconds: number | null
}

export default function DataLoading() {
  const [tables, setTables] = useState<string[]>([])
  const [selectedTable, setSelectedTable] = useState<string>('')
  const [originalFreq, setOriginalFreq] = useState<number>(10)
  const [loading, setLoading] = useState<boolean>(false)
  const [error, setError] = useState<string>('')
  const [rawData, setRawData] = useState<TableData | null>(null)
  const [preprocessedData, setPreprocessedData] = useState<PreprocessedData | null>(null)
  const [tagsData, setTagsData] = useState<TagsData | null>(null)
  const [aggregationFreq, setAggregationFreq] = useState<number | null>(null)
  const [showRawData, setShowRawData] = useState<boolean>(false)
  const [showTags, setShowTags] = useState<boolean>(false)
  const [exportData, setExportData] = useState<boolean>(false)
  const [dataSource, setDataSource] = useState<'database' | 'upload' | ''>('')
  const [tagSearch, setTagSearch] = useState<string>('')
  const [dataSearch, setDataSearch] = useState<string>('')

  const API_BASE = 'http://localhost:8000'

  // Fetch tables on component mount
  useEffect(() => {
    fetchTables()
  }, [])

  // Fetch aggregation frequency when table is selected
  useEffect(() => {
    if (selectedTable) {
      fetchAggregationFrequency()
    }
  }, [selectedTable])

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

  const fetchAggregationFrequency = async () => {
    try {
      const response = await fetch(`${API_BASE}/analytics/aggregation_frequency?table=${selectedTable}`)
      if (!response.ok) throw new Error('Failed to fetch aggregation frequency')
      const data: AggregationFrequency = await response.json()
      setAggregationFreq(data.aggregation_frequency_seconds)
    } catch (err) {
      console.error('Error fetching aggregation frequency:', err)
      setAggregationFreq(null)
    }
  }

  const loadData = async () => {
    if (!selectedTable) return

    setLoading(true)
    setError('')
    
    try {
      // Fetch raw data
      const rawResponse = await fetch(`${API_BASE}/data?table=${selectedTable}&limit=1000`)
      if (!rawResponse.ok) throw new Error('Failed to fetch raw data')
      const rawResult = await rawResponse.json()
      setRawData(rawResult)

      // Fetch preprocessed data
      const preprocessedResponse = await fetch(`${API_BASE}/data/preprocessed?table=${selectedTable}&limit=5000`)
      if (!preprocessedResponse.ok) throw new Error('Failed to fetch preprocessed data')
      const preprocessedResult = await preprocessedResponse.json()
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

  const downloadCSV = () => {
    if (!preprocessedData) return

    const headers = preprocessedData.readings.columns.join(',')
    const rows = preprocessedData.readings.rows.map(row => 
      preprocessedData.readings.columns.map(col => row[col] || '').join(',')
    ).join('\n')
    
    const csvContent = `${headers}\n${rows}`
    const blob = new Blob([csvContent], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = 'annotated_readings.csv'
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    URL.revokeObjectURL(url)
  }


  const getSearchFilteredTags = () => {
    if (!tagsData) return []
    if (!tagSearch.trim()) return tagsData.rows
    
    const searchTerm = tagSearch.toLowerCase()
    return tagsData.rows.filter(row => 
      Object.values(row).some(value => 
        value && value.toString().toLowerCase().includes(searchTerm)
      )
    )
  }

  const getFilteredDataColumns = () => {
    if (!preprocessedData) return []
    if (!dataSearch.trim()) return preprocessedData.readings.columns
    
    const searchTerm = dataSearch.toLowerCase()
    return preprocessedData.readings.columns.filter(col => 
      col.toLowerCase().includes(searchTerm)
    )
  }

  const steps = [
    { id: 1, name: 'Configure', icon: Settings, completed: selectedTable && originalFreq },
    { id: 2, name: 'Load Data', icon: Database, completed: preprocessedData !== null },
    { id: 3, name: 'Review', icon: FileText, completed: preprocessedData !== null && tagsData !== null },
    { id: 4, name: 'Export', icon: Download, completed: false }
  ]

  const currentStep = steps.findIndex(step => !step.completed) + 1 || steps.length

  return (
    <div className="space-y-8">
      {/* Header */}
      <div className="text-center">
        <h1 className="text-4xl font-bold text-white mb-4 flex items-center justify-center gap-3">
          <Zap className="text-blue-400" size={40} />
          Data Loading Wizard
        </h1>
      </div>

      {/* Progress Steps */}
      <div className="flex justify-center">
        <div className="flex items-center space-x-4">
          {steps.map((step, index) => (
            <div key={step.id} className="flex items-center">
              <div className={`
                flex items-center justify-center w-12 h-12 rounded-full border-2 transition-all duration-300
                ${step.completed 
                  ? 'bg-green-600 border-green-600 text-white' 
                  : currentStep === step.id 
                    ? 'bg-blue-600 border-blue-600 text-white animate-pulse' 
                    : 'bg-gray-700 border-gray-600 text-gray-400'
                }
              `}>
                {step.completed ? (
                  <CheckCircle size={20} />
                ) : (
                  <step.icon size={20} />
                )}
              </div>
              <div className="ml-2 hidden sm:block">
                <p className={`text-sm font-medium ${
                  step.completed ? 'text-green-400' : currentStep === step.id ? 'text-blue-400' : 'text-gray-400'
                }`}>
                  {step.name}
                </p>
              </div>
              {index < steps.length - 1 && (
                <div className={`w-16 h-0.5 ml-4 transition-colors duration-300 ${
                  step.completed ? 'bg-green-600' : 'bg-gray-600'
                }`} />
              )}
            </div>
          ))}
        </div>
      </div>

      {error && (
        <div className="bg-red-900/50 border border-red-700 rounded-xl p-4 flex items-center gap-3 animate-in slide-in-from-top duration-300">
          <AlertCircle className="text-red-400 flex-shrink-0" size={20} />
          <span className="text-red-200">{error}</span>
        </div>
      )}

      {/* Data Source Selection */}
      <div className="bg-gradient-to-r from-blue-900/30 to-purple-900/30 rounded-xl p-6 border border-blue-600/50 shadow-xl">
        <div className="text-center mb-6">
          <h2 className="text-2xl font-semibold text-white mb-2">Choose Your Data Source</h2>
          <p className="text-gray-300">Load data from the database or upload new sensor data files</p>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Database Option */}
          <div 
            onClick={() => setDataSource('database')}
            className={`bg-gray-800/80 rounded-lg p-6 border transition-all cursor-pointer transform hover:scale-105 ${
              dataSource === 'database' 
                ? 'border-blue-500 ring-2 ring-blue-500/20 bg-blue-900/20' 
                : 'border-gray-600 hover:border-blue-500'
            }`}
          >
            <div className="flex items-center gap-3 mb-4">
              <div className={`p-3 rounded-lg ${dataSource === 'database' ? 'bg-blue-500' : 'bg-blue-600'}`}>
                <Database className="text-white" size={24} />
              </div>
              <div>
                <h3 className="text-lg font-semibold text-white">LeanXcale Database</h3>
                <p className="text-gray-400 text-sm">Load existing sensor data from connected database</p>
              </div>
              {dataSource === 'database' && (
                <CheckCircle className="text-blue-400 ml-auto" size={20} />
              )}
            </div>
            <div className="space-y-2 text-sm text-gray-300">
              <p>• Access real-time and historical data</p>
              <p>• Pre-aggregated sensor readings</p>
              <p>• Automatic data preprocessing</p>
            </div>
          </div>

          {/* Upload Option - Placeholder */}
          <div 
            onClick={() => {
              // Placeholder - will be implemented by PI
              alert('Data upload feature will be implemented by the PI. This will allow users to upload CSV, Excel, and JSON files for processing.')
            }}
            className={`bg-gray-800/80 rounded-lg p-6 border border-dashed transition-all cursor-pointer transform hover:scale-105 relative ${
              dataSource === 'upload' 
                ? 'border-orange-500 ring-2 ring-orange-500/20 bg-orange-900/20' 
                : 'border-gray-500 hover:border-orange-500'
            }`}
          >
            <div className="flex items-center gap-3 mb-4">
              <div className={`p-3 rounded-lg ${dataSource === 'upload' ? 'bg-orange-500' : 'bg-orange-600'}`}>
                <CloudUpload className="text-white" size={24} />
              </div>
              <div>
                <h3 className="text-lg font-semibold text-white">Upload New Data</h3>
                <p className="text-gray-400 text-sm">Import CSV files or sensor data exports</p>
              </div>
              {dataSource === 'upload' && (
                <CheckCircle className="text-orange-400 ml-auto" size={20} />
              )}
            </div>
            
            {/* Upload Area */}
            <div className="border-2 border-dashed border-gray-600 rounded-lg p-6 text-center hover:border-orange-500 transition-colors group">
              <Upload className="mx-auto text-gray-500 group-hover:text-orange-400 mb-3 transition-colors" size={32} />
              <p className="text-gray-400 group-hover:text-gray-300 mb-2">Drop files here or click to browse</p>
              <p className="text-xs text-gray-500">Supports CSV, Excel, and JSON formats</p>
            </div>

            <div className="mt-4 space-y-2 text-sm text-gray-300">
              <p>• Upload sensor data files</p>
              <p>• Automatic format detection</p>
              <p>• Data validation and preprocessing</p>
            </div>

            {/* Coming Soon Badge */}
            <div className="absolute top-4 right-4 bg-orange-600 text-white text-xs px-3 py-1 rounded-full font-medium">
              Coming Soon
            </div>
          </div>
        </div>

        {/* Upload Instructions - Placeholder */}
        <div className="mt-6 bg-orange-900/20 border border-orange-600/30 rounded-lg p-4">
          <div className="flex items-start gap-3">
            <Plus className="text-orange-400 flex-shrink-0 mt-0.5" size={16} />
            <div>
              <p className="text-orange-200 font-medium text-sm">Data Upload Feature</p>
              <p className="text-orange-300/80 text-xs mt-1">
                This functionality will be implemented by the PI to allow users to upload and process their own sensor data files. 
                The system will support various formats and provide automatic data validation and preprocessing.
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Step 1: Configuration */}
      {dataSource === 'database' && (
        <div className={`transition-all duration-500 ${currentStep >= 1 ? 'opacity-100' : 'opacity-50'}`}>
          <div className="bg-gradient-to-r from-gray-800 to-gray-700 rounded-xl p-6 border border-gray-600 shadow-xl">
            <div className="flex items-center gap-3 mb-6">
              <div className="p-2 bg-blue-600 rounded-lg">
                <Settings className="text-white" size={24} />
              </div>
              <div>
                <h2 className="text-2xl font-semibold text-white">Step 1: Configure Database Connection</h2>
                <p className="text-gray-300">Select your machine table and set the data frequency</p>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <div className="space-y-2">
                <label className="block text-sm font-medium text-gray-300">
                  Machine Table
                </label>
                <select
                  value={selectedTable}
                  onChange={(e) => setSelectedTable(e.target.value)}
                  className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-3 text-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                >
                  <option value="">Choose a machine table...</option>
                  {tables.map(table => (
                    <option key={table} value={table}>{table}</option>
                  ))}
                </select>
                <p className="text-xs text-gray-400">Select the machine data table to analyze</p>
              </div>
              
              <div className="space-y-2">
                <label className="block text-sm font-medium text-gray-300">
                  Original Data Frequency (seconds)
                </label>
                <input
                  type="number"
                  min="1"
                  value={originalFreq}
                  onChange={(e) => setOriginalFreq(Number(e.target.value))}
                  className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-3 text-white focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                  placeholder="10"
                />
                <p className="text-xs text-gray-400">Frequency of the original sensor readings</p>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Upload Configuration Placeholder */}
      {dataSource === 'upload' && (
        <div className="transition-all duration-500">
          <div className="bg-gradient-to-r from-gray-800 to-gray-700 rounded-xl p-6 border border-gray-600 shadow-xl">
            <div className="flex items-center gap-3 mb-6">
              <div className="p-2 bg-orange-600 rounded-lg">
                <Upload className="text-white" size={24} />
              </div>
              <div>
                <h2 className="text-2xl font-semibold text-white">Step 1: Upload Configuration</h2>
                <p className="text-gray-300">Configure your data upload settings</p>
              </div>
            </div>

            <div className="bg-orange-900/20 border border-orange-600/30 rounded-lg p-6 text-center">
              <CloudUpload className="mx-auto text-orange-400 mb-4" size={48} />
              <h3 className="text-lg font-semibold text-white mb-2">Upload Feature Coming Soon</h3>
              <p className="text-orange-200 mb-4">
                The PI will implement file upload functionality including:
              </p>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm text-orange-300">
                <div className="space-y-2">
                  <p>• CSV file parsing and validation</p>
                  <p>• Excel spreadsheet support</p>
                  <p>• JSON data format handling</p>
                </div>
                <div className="space-y-2">
                  <p>• Automatic schema detection</p>
                  <p>• Data quality validation</p>
                  <p>• Progress tracking and error handling</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Step 2: Load Data */}
      <div className={`transition-all duration-500 ${currentStep >= 2 ? 'opacity-100' : 'opacity-50'}`}>
        <div className="bg-gradient-to-r from-gray-800 to-gray-700 rounded-xl p-6 border border-gray-600 shadow-xl">
          <div className="flex items-center gap-3 mb-6">
            <div className="p-2 bg-green-600 rounded-lg">
              <Database className="text-white" size={24} />
            </div>
            <div>
              <h2 className="text-2xl font-semibold text-white">Step 2: Load & Process Data</h2>
              <p className="text-gray-300">Fetch and preprocess your sensor data</p>
            </div>
          </div>

          {selectedTable && (
            <div className="space-y-4">
              <button
                onClick={loadData}
                disabled={loading || !selectedTable}
                className="w-full sm:w-auto bg-gradient-to-r from-blue-600 to-blue-700 hover:from-blue-700 hover:to-blue-800 disabled:from-gray-600 disabled:to-gray-700 text-white px-8 py-3 rounded-lg flex items-center justify-center gap-3 transition-all duration-200 transform hover:scale-105 disabled:hover:scale-100 shadow-lg"
              >
                {loading ? (
                  <>
                    <Loader2 className="animate-spin" size={20} />
                    Processing Data...
                  </>
                ) : (
                  <>
                    <TrendingUp size={20} />
                    Load & Analyze Data
                  </>
                )}
              </button>

              {aggregationFreq && (
                <div className="bg-green-900/30 border border-green-600 rounded-lg p-4 flex items-center gap-3 animate-in slide-in-from-left duration-500">
                  <CheckCircle className="text-green-400 flex-shrink-0" size={20} />
                  <div>
                    <p className="text-green-200 font-medium">
                      ✨ Aggregation frequency detected: {aggregationFreq} seconds
                    </p>
                    {originalFreq && (
                      <p className="text-green-300 text-sm">
                        Aggregation factor: {Math.round(aggregationFreq / originalFreq)}x compression
                      </p>
                    )}
                  </div>
                </div>
              )}
            </div>
          )}

          {!selectedTable && (
            <div className="text-center py-8">
              <Database className="mx-auto text-gray-500 mb-4" size={48} />
              <p className="text-gray-400">Please complete Step 1 to continue</p>
            </div>
          )}
        </div>
      </div>

      {/* Step 3: Review Data */}
      {preprocessedData && (
        <div className={`transition-all duration-500 ${currentStep >= 3 ? 'opacity-100' : 'opacity-50'}`}>
          <div className="bg-gradient-to-r from-gray-800 to-gray-700 rounded-xl p-6 border border-gray-600 shadow-xl">
            <div className="flex items-center gap-3 mb-6">
              <div className="p-2 bg-purple-600 rounded-lg">
                <FileText className="text-white" size={24} />
              </div>
              <div>
                <h2 className="text-2xl font-semibold text-white">Step 3: Review Processed Data</h2>
                <p className="text-gray-300">Examine your processed sensor data and metadata</p>
              </div>
            </div>

            {/* Data Summary Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
              <div className="bg-blue-900/30 border border-blue-600 rounded-lg p-4">
                <div className="flex items-center gap-2 mb-2">
                  <Database className="text-blue-400" size={20} />
                  <span className="text-blue-300 font-medium">Data Points</span>
                </div>
                <p className="text-2xl font-bold text-white">{preprocessedData.readings.rows.length.toLocaleString()}</p>
                <p className="text-xs text-blue-200">Total processed records</p>
              </div>

              <div className="bg-green-900/30 border border-green-600 rounded-lg p-4">
                <div className="flex items-center gap-2 mb-2">
                  <TrendingUp className="text-green-400" size={20} />
                  <span className="text-green-300 font-medium">Sensors</span>
                </div>
                <p className="text-2xl font-bold text-white">{preprocessedData.sensors.length}</p>
                <p className="text-xs text-green-200">Active sensor channels</p>
              </div>

              <div className="bg-purple-900/30 border border-purple-600 rounded-lg p-4">
                <div className="flex items-center gap-2 mb-2">
                  <Settings className="text-purple-400" size={20} />
                  <span className="text-purple-300 font-medium">Columns</span>
                </div>
                <p className="text-2xl font-bold text-white">{preprocessedData.readings.columns.length}</p>
                <p className="text-xs text-purple-200">Data attributes</p>
              </div>
            </div>

            {/* Expandable Data Table */}
            <div className="bg-gray-900/50 rounded-lg border border-gray-600">
              <button
                onClick={() => setShowRawData(!showRawData)}
                className="w-full p-4 flex items-center justify-between text-left hover:bg-gray-700/50 transition-colors rounded-lg"
              >
                <div className="flex items-center gap-3">
                  <FileText className="text-blue-400" size={20} />
                  <span className="text-lg font-semibold text-white">Processed Data Preview</span>
                  <span className="bg-blue-600 text-white text-xs px-2 py-1 rounded-full">
                    {preprocessedData.readings.rows.length} rows
                  </span>
                </div>
                {showRawData ? <ChevronDown className="text-gray-400" size={20} /> : <ChevronRight className="text-gray-400" size={20} />}
              </button>
              
              {showRawData && (
                <div className="border-t border-gray-600 p-4 animate-in slide-in-from-top duration-300">
                  {/* Column Search Input */}
                  <div className="mb-4">
                    <input
                      type="text"
                      placeholder="Search columns by name..."
                      value={dataSearch}
                      onChange={(e) => setDataSearch(e.target.value)}
                      className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent transition-all"
                    />
                  </div>

                  <div className="overflow-x-auto rounded-lg border border-gray-600">
                    <table className="w-full text-sm">
                      <thead className="bg-gray-700">
                        <tr>
                          {getFilteredDataColumns().map(col => (
                            <th key={col} className="text-left p-3 text-gray-200 font-medium border-r border-gray-600 last:border-r-0 whitespace-nowrap">
                              {col}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody className="bg-gray-800">
                        {getFilteredDataColumns().length > 0 ? (
                          preprocessedData.readings.rows.slice(0, 10).map((row, idx) => (
                            <tr key={idx} className="border-b border-gray-700 hover:bg-gray-700/50 transition-colors">
                              {getFilteredDataColumns().map(col => (
                                <td key={col} className="p-3 text-gray-200 border-r border-gray-700 last:border-r-0 whitespace-nowrap">
                                  {typeof row[col] === 'number' ? row[col].toFixed(3) : row[col] || '-'}
                                </td>
                              ))}
                            </tr>
                          ))
                        ) : (
                          <tr>
                            <td colSpan={preprocessedData.readings.columns.length} className="p-8 text-center text-gray-400">
                              {dataSearch.trim() ? (
                                <div>
                                  <p>No columns found matching "{dataSearch}"</p>
                                  <p className="text-sm mt-1">Try a different search term</p>
                                </div>
                              ) : (
                                <p>No data columns available</p>
                              )}
                            </td>
                          </tr>
                        )}
                      </tbody>
                    </table>
                  </div>
                  <div className="flex justify-between items-center mt-3 text-sm text-gray-400">
                    <span>
                      Showing first 10 of {preprocessedData.readings.rows.length} rows
                      {dataSearch.trim() && (
                        <span className="text-blue-400"> • Filtered by "{dataSearch}"</span>
                      )}
                    </span>
                    <span>
                      {getFilteredDataColumns().length}
                      {dataSearch.trim() && getFilteredDataColumns().length !== preprocessedData.readings.columns.length && (
                        <span> of {preprocessedData.readings.columns.length}</span>
                      )} columns displayed
                    </span>
                  </div>
                </div>
              )}
            </div>
          </div>
        </div>
      )}

      {/* Tags Metadata Section */}
      {tagsData && (
        <div className="bg-gradient-to-r from-gray-800 to-gray-700 rounded-xl p-6 border border-gray-600 shadow-xl">
          <div className="flex items-center gap-3 mb-6">
            <div className="p-2 bg-indigo-600 rounded-lg">
              <FileText className="text-white" size={24} />
            </div>
            <div>
              <h2 className="text-2xl font-semibold text-white">Sensor Metadata</h2>
              <p className="text-gray-300">
                Tag descriptions and equipment information 
                {selectedTable && (
                  <span className="text-blue-400 font-medium"> • Filtered for {selectedTable}</span>
                )}
              </p>
            </div>
          </div>

          {/* Auto-filtering Info */}
          {selectedTable && tagsData.rows.length > 0 && (
            <div className="mb-4 bg-indigo-900/20 border border-indigo-600/30 rounded-lg p-3">
              <div className="flex items-center gap-2 text-sm">
                <CheckCircle className="text-indigo-400" size={16} />
                <span className="text-indigo-200">
                  Automatically showing {tagsData.rows.length} sensor tags relevant to <strong>{selectedTable}</strong>
                </span>
              </div>
            </div>
          )}


          {/* Expandable Tags Table */}
          <div className="bg-gray-900/50 rounded-lg border border-gray-600">
            <button
              onClick={() => setShowTags(!showTags)}
              className="w-full p-4 flex items-center justify-between text-left hover:bg-gray-700/50 transition-colors rounded-lg"
            >
              <div className="flex items-center gap-3">
                <FileText className="text-indigo-400" size={20} />
                <span className="text-lg font-semibold text-white">Sensor Tags & Descriptions</span>
                <span className="bg-indigo-600 text-white text-xs px-2 py-1 rounded-full">
                  {tagsData.rows.length} tags {selectedTable && `for ${selectedTable}`}
                </span>
              </div>
              {showTags ? <ChevronDown className="text-gray-400" size={20} /> : <ChevronRight className="text-gray-400" size={20} />}
            </button>
            
            {showTags && (
              <div className="border-t border-gray-600 p-4 animate-in slide-in-from-top duration-300">
                {/* Search Input */}
                <div className="mb-4">
                  <input
                    type="text"
                    placeholder="Search tags, descriptions, categories..."
                    value={tagSearch}
                    onChange={(e) => setTagSearch(e.target.value)}
                    className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-2 text-white placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-indigo-500 focus:border-transparent transition-all"
                  />
                </div>

                <div className="overflow-x-auto rounded-lg border border-gray-600">
                    <table className="w-full text-sm">
                      <thead className="bg-gray-700">
                        <tr>
                          {tagsData.columns.map(col => (
                            <th key={col} className="text-left p-3 text-gray-200 font-medium border-r border-gray-600 last:border-r-0 whitespace-nowrap">
                              {col.replace('_', ' ').toUpperCase()}
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody className="bg-gray-800">
                        {getSearchFilteredTags().length > 0 ? (
                          getSearchFilteredTags().map((row, idx) => (
                            <tr key={idx} className="border-b border-gray-700 hover:bg-gray-700/50 transition-colors">
                              {tagsData.columns.map(col => (
                                <td key={col} className="p-3 text-gray-200 border-r border-gray-700 last:border-r-0 whitespace-nowrap">
                                  {row[col] || '-'}
                                </td>
                              ))}
                            </tr>
                          ))
                        ) : (
                          <tr>
                            <td colSpan={tagsData.columns.length} className="p-8 text-center text-gray-400">
                              {tagSearch.trim() ? (
                                <div>
                                  <p>No tags found matching "{tagSearch}"</p>
                                  <p className="text-sm mt-1">Try a different search term</p>
                                </div>
                              ) : (
                                <p>No sensor tags available</p>
                              )}
                            </td>
                          </tr>
                        )}
                      </tbody>
                    </table>
                  </div>
                  <div className="mt-3 flex justify-between items-center text-sm text-gray-400">
                    <span>
                      Showing {getSearchFilteredTags().length} 
                      {tagSearch.trim() && getSearchFilteredTags().length !== tagsData.rows.length && (
                        <span> of {tagsData.rows.length}</span>
                      )} sensor tags for <strong className="text-gray-300">{selectedTable}</strong>
                      {tagSearch.trim() && (
                        <span className="text-indigo-400"> • Filtered by "{tagSearch}"</span>
                      )}
                    </span>
                    <span>{tagsData.columns.length} attributes per tag</span>
                  </div>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Step 4: Export & Finalize */}
      {preprocessedData && (
        <div className={`transition-all duration-500 ${currentStep >= 4 ? 'opacity-100' : 'opacity-50'}`}>
          <div className="bg-gradient-to-r from-gray-800 to-gray-700 rounded-xl p-6 border border-gray-600 shadow-xl">
            <div className="flex items-center gap-3 mb-6">
              <div className="p-2 bg-orange-600 rounded-lg">
                <Download className="text-white" size={24} />
              </div>
              <div>
                <h2 className="text-2xl font-semibold text-white">Step 4: Export & Next Steps</h2>
                <p className="text-gray-300">Download your data or proceed to analysis</p>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Export Option */}
              <div className="space-y-4">
                <div className="flex items-center gap-3">
                  <input
                    type="checkbox"
                    id="exportData"
                    checked={exportData}
                    onChange={(e) => setExportData(e.target.checked)}
                    className="w-5 h-5 text-orange-600 bg-gray-700 border-gray-600 rounded focus:ring-orange-500"
                  />
                  <label htmlFor="exportData" className="text-gray-200 font-medium">
                    Export processed dataset
                  </label>
                </div>
                
                {exportData && (
                  <button
                    onClick={downloadCSV}
                    className="w-full bg-gradient-to-r from-orange-600 to-orange-700 hover:from-orange-700 hover:to-orange-800 text-white px-6 py-3 rounded-lg flex items-center justify-center gap-3 transition-all duration-200 transform hover:scale-105 shadow-lg"
                  >
                    <Download size={20} />
                    Download CSV Dataset
                  </button>
                )}
              </div>

              {/* Next Steps */}
              <div className="bg-green-900/30 border border-green-600 rounded-lg p-4">
                <div className="flex items-center gap-2 mb-3">
                  <CheckCircle className="text-green-400" size={20} />
                  <span className="text-green-300 font-medium">Ready for Analysis!</span>
                </div>
                <p className="text-green-200 text-sm mb-3">
                  Your data has been successfully processed and is ready for visualization and quality analysis.
                </p>
                <div className="space-y-2 text-sm">
                  <p className="text-green-300">• Navigate to <strong>Data Visualization</strong> to explore trends</p>
                  <p className="text-green-300">• Use <strong>Missing Values Analysis</strong> to identify gaps</p>
                  <p className="text-green-300">• Run <strong>Data Quality Assessment</strong> for comprehensive metrics</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Helper Messages */}
      {!selectedTable && (
        <div className="text-center py-12">
          <Database className="mx-auto text-gray-500 mb-4" size={64} />
          <h3 className="text-xl font-semibold text-gray-300 mb-2">Get Started</h3>
          <p className="text-gray-400">Select a machine table above to begin loading your IIoT sensor data</p>
        </div>
      )}

      {selectedTable && !preprocessedData && !loading && (
        <div className="text-center py-12">
          <TrendingUp className="mx-auto text-blue-400 mb-4" size={64} />
          <h3 className="text-xl font-semibold text-white mb-2">Ready to Load</h3>
          <p className="text-gray-300">Click "Load & Analyze Data" to process your selected table</p>
        </div>
      )}
    </div>
  )
}


