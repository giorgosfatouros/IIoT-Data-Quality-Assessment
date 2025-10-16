import { NavLink, Outlet } from 'react-router-dom'
import { useState } from 'react'
import { Menu, X } from 'lucide-react'
import logoAzul from './assets/logo-azul.svg'

function Header({ onMenuToggle }: { onMenuToggle: () => void }) {
  return (
    <header className="bg-gray-800 border-b border-gray-700 flex-shrink-0">
      <div className="px-4 md:px-6 h-16 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <button
            onClick={onMenuToggle}
            className="md:hidden p-2 rounded-md text-gray-400 hover:text-white hover:bg-gray-700 transition-colors"
          >
            <Menu size={20} />
          </button>
          <div className="h-8 w-8 rounded bg-blue-600" />
          <span className="font-semibold text-white text-lg hidden sm:block">IIoT Data Quality Dashboard</span>
          <span className="font-semibold text-white text-base sm:hidden">IIoT DQ</span>
        </div>
        <div className="text-xs text-gray-400">v0.1</div>
      </div>
    </header>
  )
}

function Sidebar({ isOpen, onClose }: { isOpen: boolean; onClose: () => void }) {
  const items = [
    { to: '/', label: 'Home' },
    { to: '/data-loading', label: 'Data Loading' },
    { to: '/data-visualization', label: 'Data Visualization' },
    { to: '/missing-values', label: 'Missing Values' },
    { to: '/invalid-values', label: 'Invalid Values' },
    { to: '/data-quality', label: 'Data Quality' },
    { to: '/dqa-agent', label: 'DQA Agent' },
  ]
  
  return (
    <>
      {/* Mobile backdrop */}
      {isOpen && (
        <div 
          className="fixed inset-0 bg-black/50 z-40 md:hidden"
          onClick={onClose}
        />
      )}
      
      {/* Sidebar */}
      <aside className={`
        fixed md:relative top-0 left-0 z-50 w-64 h-full bg-gray-800 border-r border-gray-700 flex-shrink-0 transform transition-transform duration-200 ease-in-out
        ${isOpen ? 'translate-x-0' : '-translate-x-full'}
        md:translate-x-0 md:block
      `}>
        <div className="p-4 border-b border-gray-700 flex items-center justify-between">
          <img 
            src={logoAzul} 
            alt="FAME Logo" 
            className="h-8 w-auto"
          />
          <button
            onClick={onClose}
            className="md:hidden p-1 rounded text-gray-400 hover:text-white hover:bg-gray-700 transition-colors"
          >
            <X size={18} />
          </button>
        </div>
        <nav className="p-2 space-y-1">
          {items.map(it => (
            <NavLink
              key={it.to}
              to={it.to}
              end={it.to === '/'}
              onClick={() => onClose()}
              className={({ isActive }) =>
                `block px-3 py-2 rounded transition-colors hover:bg-gray-700 text-gray-300 hover:text-white ${
                  isActive ? 'bg-gray-700 text-white font-medium' : ''
                }`
              }
            >
              {it.label}
            </NavLink>
          ))}
        </nav>
      </aside>
    </>
  )
}

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(false)
  
  return (
    <div className="h-screen bg-gradient-to-br from-gray-900 to-gray-800 text-gray-100 flex flex-col overflow-hidden">
      <Header onMenuToggle={() => setSidebarOpen(true)} />
      <div className="flex flex-1 overflow-hidden relative">
        <Sidebar isOpen={sidebarOpen} onClose={() => setSidebarOpen(false)} />
        <main className="flex-1 overflow-y-auto p-4 md:p-6">
          <Outlet />
        </main>
      </div>
    </div>
  )
}

export default App


