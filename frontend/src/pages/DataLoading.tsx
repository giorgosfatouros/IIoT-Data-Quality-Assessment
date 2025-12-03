import { useState, useEffect, useCallback } from 'react'
import { ChevronDown, ChevronRight, Download, Loader2, AlertCircle, CheckCircle, Database, Settings, FileText, Zap, TrendingUp, Upload, CloudUpload, Plus, X, FileCheck } from 'lucide-react'
import Footer from '../components/Footer'

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
  
  // Upload-related state
  const [uploadDataFile, setUploadDataFile] = useState<File | null>(null)
  const [uploadTagsFile, setUploadTagsFile] = useState<File | null>(null)
  const [uploadTableName, setUploadTableName] = useState<string>('')
  const [uploadMachineType, setUploadMachineType] = useState<string>('AUTO')
  const [uploadValidation, setUploadValidation] = useState<any>(null)
  const [isValidating, setIsValidating] = useState<boolean>(false)
  const [isUploading, setIsUploading] = useState<boolean>(false)
  const [uploadJobId, setUploadJobId] = useState<string>('')
  const [uploadProgress, setUploadProgress] = useState<number>(0)
  const [uploadStatus, setUploadStatus] = useState<string>('')
  const [importAvailable, setImportAvailable] = useState<boolean>(false)
  const [importHealthMessage, setImportHealthMessage] = useState<string>('')
  const [selectedSensors, setSelectedSensors] = useState<Set<string>>(new Set())

  const API_BASE = 'http://localhost:8000'

  // Fetch tables on component mount
  useEffect(() => {
    fetchTables()
    checkImportHealth()
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
      // Use all machine groups from aggregated_insights (no filtering needed)
      // The /tables endpoint already returns distinct machine_group values
      setTables(allTables)
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

  // Check if import service is available
  const checkImportHealth = async () => {
    try {
      const response = await fetch(`${API_BASE}/import/health`)
      const health = await response.json()
      setImportAvailable(health.docker_available && health.importer_image_available)
      setImportHealthMessage(health.message)
    } catch (err) {
      console.error('Import health check failed:', err)
      setImportAvailable(false)
      setImportHealthMessage('Import service unavailable')
    }
  }

  // Handle file selection
  const handleDataFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setUploadDataFile(e.target.files[0])
      setUploadValidation(null)
    }
  }

  const handleTagsFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setUploadTagsFile(e.target.files[0])
      setUploadValidation(null)
    }
  }

  // Validate files before upload
  const validateUploadFiles = async () => {
    if (!uploadDataFile || !uploadTagsFile) {
      setError('Please select both data and tags files')
      return
    }

    setIsValidating(true)
    setError('')

    try {
      const formData = new FormData()
      formData.append('data_file', uploadDataFile)
      formData.append('tags_file', uploadTagsFile)

      const response = await fetch(`${API_BASE}/import/validate`, {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        throw new Error('Validation request failed')
      }

      const validation = await response.json()
      setUploadValidation(validation)

      if (!validation.can_proceed) {
        const errors = [
          ...validation.data_file.errors,
          ...validation.tags_file.errors
        ]
        setError(`Validation failed: ${errors.join(', ')}`)
      } else {
        // Set suggested table name if not already set
        if (!uploadTableName && validation.suggested_table_name) {
          setUploadTableName(validation.suggested_table_name)
        }
        // Auto-select all sensors by default
        if (validation.available_sensors && validation.available_sensors.length > 0) {
          setSelectedSensors(new Set(validation.available_sensors))
        }
      }
    } catch (err) {
      setError('Validation error: ' + (err as Error).message)
    } finally {
      setIsValidating(false)
    }
  }

  // Upload and start import
  const startImport = async () => {
    if (!uploadDataFile || !uploadTagsFile || !uploadTableName) {
      setError('Please provide all required information')
      return
    }

    if (uploadValidation && !uploadValidation.can_proceed) {
      setError('Please fix validation errors before importing')
      return
    }

    setIsUploading(true)
    setError('')
    setUploadProgress(0)

    try {
      const formData = new FormData()
      formData.append('data_file', uploadDataFile)
      formData.append('tags_file', uploadTagsFile)
      formData.append('table_name', uploadTableName)
      formData.append('machine_type', uploadMachineType)
      
      // Add selected sensors if any are selected (otherwise import all)
      if (selectedSensors.size > 0 && selectedSensors.size < (uploadValidation?.available_sensors?.length || Infinity)) {
        formData.append('selected_sensors', Array.from(selectedSensors).join(','))
      }

      const response = await fetch(`${API_BASE}/import/upload`, {
        method: 'POST',
        body: formData
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Upload failed')
      }

      const job = await response.json()
      setUploadJobId(job.job_id)
      setUploadStatus(job.status)

      // Start polling for status
      pollImportStatus(job.job_id)
    } catch (err) {
      setError('Upload error: ' + (err as Error).message)
      setIsUploading(false)
    }
  }

  // Poll import status
  const pollImportStatus = async (jobId: string) => {
    try {
      const response = await fetch(`${API_BASE}/import/status/${jobId}`)
      
      if (!response.ok) {
        throw new Error('Status check failed')
      }

      const status = await response.json()
      setUploadStatus(status.status)
      setUploadProgress(status.progress_percentage || 0)

      if (status.status === 'completed') {
        setIsUploading(false)
        setUploadProgress(100)
        // Refresh tables list
        fetchTables()
        // Optionally load the new table
        if (status.table_name) {
          setSelectedTable(status.table_name)
        }
      } else if (status.status === 'failed') {
        setIsUploading(false)
        setError(status.error_message || 'Import failed')
      } else if (status.status === 'cancelled') {
        setIsUploading(false)
        setError('Import was cancelled')
      } else {
        // Continue polling
        setTimeout(() => pollImportStatus(jobId), 3000)
      }
    } catch (err) {
      setError('Status check error: ' + (err as Error).message)
      setIsUploading(false)
    }
  }

  // Reset upload form
  const resetUploadForm = () => {
    setUploadDataFile(null)
    setUploadTagsFile(null)
    setUploadTableName('')
    setUploadMachineType('AUTO')
    setUploadValidation(null)
    setUploadJobId('')
    setUploadProgress(0)
    setUploadStatus('')
    setSelectedSensors(new Set())
  }
  
  // Toggle sensor selection
  const toggleSensor = (sensor: string) => {
    const newSelected = new Set(selectedSensors)
    if (newSelected.has(sensor)) {
      newSelected.delete(sensor)
    } else {
      newSelected.add(sensor)
    }
    setSelectedSensors(newSelected)
  }
  
  // Select/deselect all sensors
  const toggleAllSensors = () => {
    if (uploadValidation?.available_sensors) {
      if (selectedSensors.size === uploadValidation.available_sensors.length) {
        setSelectedSensors(new Set())
      } else {
        setSelectedSensors(new Set(uploadValidation.available_sensors))
      }
    }
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
                <h3 className="text-lg font-semibold text-white">Database</h3>
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

          {/* Upload Option - Now Functional */}
          <div 
            onClick={() => setDataSource('upload')}
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
                <p className="text-gray-400 text-sm">Import CSV files via moh-importer</p>
              </div>
              {dataSource === 'upload' && (
                <CheckCircle className="text-orange-400 ml-auto" size={20} />
              )}
            </div>
            
            <div className="mt-4 space-y-2 text-sm text-gray-300">
              <p>• Upload sensor data CSV files</p>
              <p>• Automatic machine type detection</p>
              <p>• Real-time validation and import</p>
            </div>

            {/* Status Badge */}
            {importAvailable ? (
              <div className="absolute top-4 right-4 bg-green-600 text-white text-xs px-3 py-1 rounded-full font-medium flex items-center gap-1">
                <CheckCircle size={12} />
                Ready
              </div>
            ) : (
              <div className="absolute top-4 right-4 bg-yellow-600 text-white text-xs px-3 py-1 rounded-full font-medium flex items-center gap-1">
                <AlertCircle size={12} />
                Setup Required
              </div>
            )}
          </div>
        </div>

        {/* Import Health Status */}
        {!importAvailable && (
          <div className="mt-6 bg-yellow-900/20 border border-yellow-600/30 rounded-lg p-4">
            <div className="flex items-start gap-3">
              <AlertCircle className="text-yellow-400 flex-shrink-0 mt-0.5" size={16} />
              <div>
                <p className="text-yellow-200 font-medium text-sm">Import Service Status</p>
                <p className="text-yellow-300/80 text-xs mt-1">
                  {importHealthMessage || 'Docker or moh-importer image not available. Please ensure Docker is running and the image is built.'}
                </p>
                <p className="text-yellow-300/80 text-xs mt-2">
                  Build command: <code className="bg-black/30 px-2 py-1 rounded">cd /home/george/moh-importer-main && docker build -t moh-importer:latest .</code>
                </p>
              </div>
            </div>
          </div>
        )}
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

      {/* Upload Configuration - Now Functional */}
      {dataSource === 'upload' && importAvailable && (
        <div className="transition-all duration-500">
          <div className="bg-gradient-to-r from-gray-800 to-gray-700 rounded-xl p-6 border border-gray-600 shadow-xl">
            <div className="flex items-center gap-3 mb-6">
              <div className="p-2 bg-orange-600 rounded-lg">
                <Upload className="text-white" size={24} />
              </div>
              <div>
                <h2 className="text-2xl font-semibold text-white">Step 1: Upload and Import Data</h2>
                <p className="text-gray-300">Select CSV files and configure import settings</p>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Data File Upload */}
              <div className="space-y-3">
                <label className="block text-sm font-medium text-gray-300">
                  Sensor Data CSV File *
                </label>
                <div className="relative">
                  <input
                    type="file"
                    accept=".csv"
                    onChange={handleDataFileChange}
                    className="hidden"
                    id="data-file-input"
                  />
                  <label
                    htmlFor="data-file-input"
                    className="flex items-center justify-center gap-2 w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-3 text-white cursor-pointer hover:bg-gray-600 transition-all"
                  >
                    {uploadDataFile ? (
                      <>
                        <FileCheck size={20} className="text-green-400" />
                        <span className="text-sm truncate">{uploadDataFile.name}</span>
                        <X
                          size={16}
                          className="ml-auto text-gray-400 hover:text-red-400"
                          onClick={(e) => {
                            e.preventDefault()
                            setUploadDataFile(null)
                          }}
                        />
                      </>
                    ) : (
                      <>
                        <CloudUpload size={20} />
                        <span>Select Data File</span>
                      </>
                    )}
                  </label>
                </div>
                <p className="text-xs text-gray-400">
                  CSV with timestamp and sensor readings
                </p>
              </div>

              {/* Tags File Upload */}
              <div className="space-y-3">
                <label className="block text-sm font-medium text-gray-300">
                  Tags/Thresholds CSV File *
                </label>
                <div className="relative">
                  <input
                    type="file"
                    accept=".csv"
                    onChange={handleTagsFileChange}
                    className="hidden"
                    id="tags-file-input"
                  />
                  <label
                    htmlFor="tags-file-input"
                    className="flex items-center justify-center gap-2 w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-3 text-white cursor-pointer hover:bg-gray-600 transition-all"
                  >
                    {uploadTagsFile ? (
                      <>
                        <FileCheck size={20} className="text-green-400" />
                        <span className="text-sm truncate">{uploadTagsFile.name}</span>
                        <X
                          size={16}
                          className="ml-auto text-gray-400 hover:text-red-400"
                          onClick={(e) => {
                            e.preventDefault()
                            setUploadTagsFile(null)
                          }}
                        />
                      </>
                    ) : (
                      <>
                        <CloudUpload size={20} />
                        <span>Select Tags File</span>
                      </>
                    )}
                  </label>
                </div>
                <p className="text-xs text-gray-400">
                  CSV with sensor tags and validation rules
                </p>
              </div>
            </div>

            {/* Configuration Options */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-6">
              <div className="space-y-2">
                <label className="block text-sm font-medium text-gray-300">
                  Target Table Name *
                </label>
                <input
                  type="text"
                  value={uploadTableName}
                  onChange={(e) => setUploadTableName(e.target.value.toUpperCase())}
                  placeholder="e.g., KT2201"
                  className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-3 text-white focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-transparent transition-all"
                />
                <p className="text-xs text-gray-400">
                  {uploadValidation?.suggested_table_name && 
                    `Suggested: ${uploadValidation.suggested_table_name}`
                  }
                </p>
              </div>

              <div className="space-y-2">
                <label className="block text-sm font-medium text-gray-300">
                  Machine Type
                </label>
                <select
                  value={uploadMachineType}
                  onChange={(e) => setUploadMachineType(e.target.value)}
                  className="w-full bg-gray-700 border border-gray-600 rounded-lg px-4 py-3 text-white focus:outline-none focus:ring-2 focus:ring-orange-500 focus:border-transparent transition-all"
                >
                  <option value="AUTO">Auto-detect</option>
                  <option value="KT2201">KT2201</option>
                  <option value="K3301">K3301</option>
                  <option value="K5700">K5700</option>
                </select>
                <p className="text-xs text-gray-400">
                  {uploadValidation?.suggested_machine_type &&
                    `Detected: ${uploadValidation.suggested_machine_type}`
                  }
                </p>
              </div>
            </div>

            {/* Action Buttons */}
            <div className="flex gap-4 mt-6">
              <button
                onClick={validateUploadFiles}
                disabled={!uploadDataFile || !uploadTagsFile || isValidating}
                className="flex-1 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 text-white px-6 py-3 rounded-lg flex items-center justify-center gap-2 transition-all disabled:cursor-not-allowed"
              >
                {isValidating ? (
                  <>
                    <Loader2 className="animate-spin" size={20} />
                    Validating...
                  </>
                ) : (
                  <>
                    <FileCheck size={20} />
                    Validate Files
                  </>
                )}
              </button>

              <button
                onClick={startImport}
                disabled={!uploadDataFile || !uploadTagsFile || !uploadTableName || isUploading || (uploadValidation && !uploadValidation.can_proceed)}
                className="flex-1 bg-orange-600 hover:bg-orange-700 disabled:bg-gray-600 text-white px-6 py-3 rounded-lg flex items-center justify-center gap-2 transition-all disabled:cursor-not-allowed"
              >
                {isUploading ? (
                  <>
                    <Loader2 className="animate-spin" size={20} />
                    Importing... {uploadProgress.toFixed(0)}%
                  </>
                ) : (
                  <>
                    <CloudUpload size={20} />
                    Start Import
                  </>
                )}
              </button>

              {(uploadDataFile || uploadTagsFile) && (
                <button
                  onClick={resetUploadForm}
                  disabled={isUploading}
                  className="bg-gray-600 hover:bg-gray-700 disabled:bg-gray-600 text-white px-6 py-3 rounded-lg flex items-center justify-center gap-2 transition-all disabled:cursor-not-allowed"
                >
                  <X size={20} />
                  Reset
                </button>
              )}
            </div>

            {/* Validation Results */}
            {uploadValidation && (
              <div className="mt-6 space-y-4">
                {/* Data File Validation */}
                <div className={`rounded-lg p-4 border ${
                  uploadValidation.data_file.is_valid
                    ? 'bg-green-900/20 border-green-600'
                    : 'bg-red-900/20 border-red-600'
                }`}>
                  <div className="flex items-center gap-2 mb-2">
                    {uploadValidation.data_file.is_valid ? (
                      <CheckCircle className="text-green-400" size={20} />
                    ) : (
                      <AlertCircle className="text-red-400" size={20} />
                    )}
                    <h3 className="font-semibold text-white">
                      Data File: {uploadValidation.data_file.filename}
                    </h3>
                  </div>
                  <div className="text-sm space-y-1">
                    <p className="text-gray-300">
                      Rows: {uploadValidation.data_file.row_count?.toLocaleString() || 'N/A'} | 
                      Columns: {uploadValidation.data_file.column_count || 'N/A'}
                    </p>
                    {uploadValidation.data_file.sensor_tags && uploadValidation.data_file.sensor_tags.length > 0 && (
                      <div className="mt-3">
                        <p className="text-blue-300 font-medium mb-2">
                          Identified Sensors ({uploadValidation.data_file.sensor_tags.length}):
                        </p>
                        <div className="flex flex-wrap gap-2 max-h-32 overflow-y-auto bg-gray-800/50 rounded p-2">
                          {uploadValidation.data_file.sensor_tags.map((tag: string, i: number) => (
                            <span
                              key={i}
                              className="inline-block bg-blue-600/30 text-blue-200 px-2 py-1 rounded text-xs border border-blue-500/30"
                            >
                              {tag}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                    {uploadValidation.data_file.errors.length > 0 && (
                      <div className="text-red-300 mt-2">
                        {uploadValidation.data_file.errors.map((err: string, i: number) => (
                          <p key={i}>❌ {err}</p>
                        ))}
                      </div>
                    )}
                    {uploadValidation.data_file.warnings.length > 0 && (
                      <div className="text-yellow-300 mt-2">
                        {uploadValidation.data_file.warnings.map((warn: string, i: number) => (
                          <p key={i}>⚠️ {warn}</p>
                        ))}
                      </div>
                    )}
                  </div>
                </div>

                {/* Tags File Validation */}
                <div className={`rounded-lg p-4 border ${
                  uploadValidation.tags_file.is_valid
                    ? 'bg-green-900/20 border-green-600'
                    : 'bg-red-900/20 border-red-600'
                }`}>
                  <div className="flex items-center gap-2 mb-2">
                    {uploadValidation.tags_file.is_valid ? (
                      <CheckCircle className="text-green-400" size={20} />
                    ) : (
                      <AlertCircle className="text-red-400" size={20} />
                    )}
                    <h3 className="font-semibold text-white">
                      Tags File: {uploadValidation.tags_file.filename}
                    </h3>
                  </div>
                  <div className="text-sm space-y-1">
                    <p className="text-gray-300">
                      Tags: {uploadValidation.tags_file.tag_count?.toLocaleString() || 'N/A'}
                    </p>
                    {uploadValidation.tags_file.errors.length > 0 && (
                      <div className="text-red-300">
                        {uploadValidation.tags_file.errors.map((err: string, i: number) => (
                          <p key={i}>❌ {err}</p>
                        ))}
                      </div>
                    )}
                    {uploadValidation.tags_file.warnings.length > 0 && (
                      <div className="text-yellow-300">
                        {uploadValidation.tags_file.warnings.map((warn: string, i: number) => (
                          <p key={i}>⚠️ {warn}</p>
                        ))}
                      </div>
                    )}
                  </div>
                </div>

                {/* Sensor Selection */}
                {uploadValidation.can_proceed && uploadValidation.available_sensors && uploadValidation.available_sensors.length > 0 && (
                  <div className="bg-blue-900/20 border border-blue-600 rounded-lg p-4">
                    <div className="flex items-center justify-between mb-3">
                      <div className="flex items-center gap-2">
                        <Settings className="text-blue-400" size={20} />
                        <h3 className="font-semibold text-white">
                          Select Sensors to Import
                        </h3>
                      </div>
                      <button
                        onClick={toggleAllSensors}
                        className="text-sm text-blue-300 hover:text-blue-200 underline"
                      >
                        {selectedSensors.size === uploadValidation.available_sensors.length ? 'Deselect All' : 'Select All'}
                      </button>
                    </div>
                    <p className="text-sm text-blue-300 mb-3">
                      {selectedSensors.size} of {uploadValidation.available_sensors.length} sensors selected
                    </p>
                    <div className="max-h-64 overflow-y-auto bg-gray-800/50 rounded p-3 space-y-2">
                      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-2">
                        {uploadValidation.available_sensors.map((sensor: string) => (
                          <label
                            key={sensor}
                            className="flex items-center gap-2 p-2 bg-gray-700/50 hover:bg-gray-700 rounded cursor-pointer transition-colors"
                          >
                            <input
                              type="checkbox"
                              checked={selectedSensors.has(sensor)}
                              onChange={() => toggleSensor(sensor)}
                              className="w-4 h-4 text-blue-600 bg-gray-700 border-gray-600 rounded focus:ring-blue-500"
                            />
                            <span className="text-sm text-gray-200">{sensor}</span>
                          </label>
                        ))}
                      </div>
                    </div>
                  </div>
                )}

                {/* Overall Status */}
                {uploadValidation.can_proceed && (
                  <div className="bg-green-900/20 border border-green-600 rounded-lg p-4">
                    <div className="flex items-center gap-2">
                      <CheckCircle className="text-green-400" size={20} />
                      <p className="text-green-200 font-medium">
                        ✅ Validation passed! Ready to import.
                        {selectedSensors.size > 0 && selectedSensors.size < (uploadValidation?.available_sensors?.length || 0) && (
                          <span className="ml-2 text-green-300 text-sm">
                            ({selectedSensors.size} sensors selected)
                          </span>
                        )}
                      </p>
                    </div>
                  </div>
                )}
              </div>
            )}

            {/* Import Progress */}
            {isUploading && uploadJobId && (
              <div className="mt-6 bg-blue-900/20 border border-blue-600 rounded-lg p-4">
                <div className="flex items-center gap-3 mb-3">
                  <Loader2 className="animate-spin text-blue-400" size={20} />
                  <div>
                    <p className="text-blue-200 font-medium">Import in Progress</p>
                    <p className="text-blue-300 text-sm">Job ID: {uploadJobId}</p>
                    <p className="text-blue-300 text-sm">Status: {uploadStatus}</p>
                  </div>
                </div>
                <div className="w-full bg-gray-700 rounded-full h-2">
                  <div
                    className="bg-blue-500 h-2 rounded-full transition-all duration-300"
                    style={{ width: `${uploadProgress}%` }}
                  />
                </div>
                <p className="text-center text-blue-200 text-sm mt-2">
                  {uploadProgress.toFixed(1)}% Complete
                </p>
              </div>
            )}
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
      
      <Footer />
    </div>
  )
}


