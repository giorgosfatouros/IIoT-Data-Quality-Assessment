import { useState } from 'react'
import { Layers } from 'lucide-react'

interface SensorInfo {
  name: string
  category?: string
  description?: string
  unit?: string
}

interface SensorSelectorProps {
  sensors: string[]
  selectedSensors: string[]
  onSelectionChange: (sensors: string[]) => void
  getSensorInfo?: (sensor: string) => SensorInfo | null
  accentColor?: 'blue' | 'red' | 'orange' | 'green'
}

interface TooltipState {
  show: boolean
  content: SensorInfo | null
  x: number
  y: number
}

export default function SensorSelector({
  sensors,
  selectedSensors,
  onSelectionChange,
  getSensorInfo,
  accentColor = 'blue'
}: SensorSelectorProps) {
  const [tooltip, setTooltip] = useState<TooltipState>({ show: false, content: null, x: 0, y: 0 })

  const handleMouseEnter = (event: React.MouseEvent<HTMLButtonElement>, info: SensorInfo | null) => {
    if (!info || (!info.description && !info.unit)) return
    
    const rect = event.currentTarget.getBoundingClientRect()
    setTooltip({
      show: true,
      content: info,
      x: rect.left + rect.width / 2,
      y: rect.top - 10
    })
  }

  const handleMouseLeave = () => {
    setTooltip({ show: false, content: null, x: 0, y: 0 })
  }

  const colorClasses = {
    blue: {
      selected: 'bg-blue-600 text-white border-blue-500',
      selectAllBtn: 'bg-blue-600 hover:bg-blue-700'
    },
    red: {
      selected: 'bg-red-600 text-white border-red-500',
      selectAllBtn: 'bg-red-600 hover:bg-red-700'
    },
    orange: {
      selected: 'bg-orange-600 text-white border-orange-500',
      selectAllBtn: 'bg-orange-600 hover:bg-orange-700'
    },
    green: {
      selected: 'bg-green-600 text-white border-green-500',
      selectAllBtn: 'bg-green-600 hover:bg-green-700'
    }
  }

  const colors = colorClasses[accentColor]

  const toggleSensor = (sensor: string) => {
    if (selectedSensors.includes(sensor)) {
      onSelectionChange(selectedSensors.filter(s => s !== sensor))
    } else {
      onSelectionChange([...selectedSensors, sensor])
    }
  }

  const selectAll = () => {
    onSelectionChange(sensors)
  }

  const clearAll = () => {
    onSelectionChange([])
  }

  return (
    <div className="mt-6 pt-6 border-t border-gray-700">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <label className="text-sm font-medium text-gray-300 flex items-center gap-2">
          <Layers size={16} />
          Select Sensors (Optional - leave empty to analyze all)
        </label>
        <div className="flex gap-2">
          <button
            onClick={selectAll}
            className={`text-xs ${colors.selectAllBtn} text-white px-3 py-1 rounded transition-colors`}
          >
            Select All
          </button>
          <button
            onClick={clearAll}
            className="text-xs bg-gray-600 hover:bg-gray-700 text-white px-3 py-1 rounded transition-colors"
          >
            Clear
          </button>
        </div>
      </div>

      {/* Sensor Chips */}
      <div className="flex flex-wrap gap-2 max-h-64 overflow-y-auto p-3 bg-gray-900/30 rounded-lg border border-gray-600">
        {sensors.map(sensor => {
          const isSelected = selectedSensors.includes(sensor)
          const info = getSensorInfo ? getSensorInfo(sensor) : null

          return (
            <button
              key={sensor}
              onClick={() => toggleSensor(sensor)}
              onMouseEnter={(e) => handleMouseEnter(e, info)}
              onMouseLeave={handleMouseLeave}
              className={`px-3 py-2 rounded-lg text-sm transition-all border ${
                isSelected
                  ? colors.selected
                  : 'bg-gray-700 text-gray-300 hover:bg-gray-600 border-gray-600'
              }`}
            >
              <div className="flex flex-col items-start">
                <span className="font-medium">{sensor}</span>
                {info?.category && (
                  <span className={`text-xs ${isSelected ? 'text-white/70' : 'text-gray-400'}`}>
                    {info.category}
                  </span>
                )}
              </div>
            </button>
          )
        })}
      </div>

      {/* Fixed position tooltip */}
      {tooltip.show && tooltip.content && (
        <div
          className="fixed z-[9999] pointer-events-none"
          style={{
            left: `${tooltip.x}px`,
            top: `${tooltip.y}px`,
            transform: 'translate(-50%, -100%)'
          }}
        >
          <div className="bg-gray-900 border border-gray-600 rounded-lg p-3 shadow-2xl text-xs text-left max-w-xs">
            <div className="text-white font-semibold mb-1">{tooltip.content.name}</div>
            {tooltip.content.description && (
              <div className="text-gray-300 mb-1 whitespace-normal">{tooltip.content.description}</div>
            )}
            {tooltip.content.category && (
              <div className="text-gray-400">Category: {tooltip.content.category}</div>
            )}
            {tooltip.content.unit && (
              <div className="text-gray-400">Unit: {tooltip.content.unit}</div>
            )}
            {/* Arrow */}
            <div className="absolute top-full left-1/2 transform -translate-x-1/2 -mt-px">
              <div className="border-4 border-transparent border-t-gray-600"></div>
            </div>
          </div>
        </div>
      )}

      {/* Selection Summary */}
      <p className="text-xs text-gray-400 mt-2">
        {selectedSensors.length > 0
          ? `${selectedSensors.length} sensor${selectedSensors.length !== 1 ? 's' : ''} selected`
          : 'No sensors selected - will analyze all sensors'}
      </p>
    </div>
  )
}
