import React from 'react'
import { createRoot } from 'react-dom/client'
import { createBrowserRouter, RouterProvider } from 'react-router-dom'
import App from './App'
import './styles.css'

import Home from './pages/Home'
import DataLoading from './pages/DataLoading'
import DataVisualization from './pages/DataVisualization'
import MissingValues from './pages/MissingValues'
import InvalidValues from './pages/InvalidValues'
import DataQuality from './pages/DataQuality'

const router = createBrowserRouter([
  {
    path: '/',
    element: <App />,
    children: [
      { index: true, element: <Home /> },
      { path: 'data-loading', element: <DataLoading /> },
      { path: 'data-visualization', element: <DataVisualization /> },
      { path: 'missing-values', element: <MissingValues /> },
      { path: 'invalid-values', element: <InvalidValues /> },
      { path: 'data-quality', element: <DataQuality /> },
    ],
  },
])

createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <RouterProvider router={router} />
  </React.StrictMode>
)


