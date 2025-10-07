import { ReactNode } from 'react'
import { Link } from 'react-router-dom'

type CardProps = {
  title: string
  description?: string
  icon?: ReactNode
  to?: string
  actions?: ReactNode
}

export default function Card({ title, description, icon, to, actions }: CardProps) {
  const Inner = (
    <div className="group h-full rounded-xl border border-gray-700 bg-gray-800/80 backdrop-blur shadow-lg hover:shadow-xl hover:border-gray-600 transition-all duration-200 p-6 flex flex-col">
      <div className="flex items-start gap-4 flex-1">
        {icon && (
          <div className="h-12 w-12 flex items-center justify-center rounded-lg bg-blue-600/20 text-blue-400 flex-shrink-0">
            {icon}
          </div>
        )}
        <div className="flex-1 min-w-0">
          <h3 className="text-lg font-semibold text-white mb-2 group-hover:text-blue-300 transition-colors">{title}</h3>
          {description && <p className="text-sm text-gray-400 leading-relaxed">{description}</p>}
        </div>
      </div>
      {actions && <div className="mt-4 pt-4 border-t border-gray-700">{actions}</div>}
    </div>
  )

  if (to) {
    return (
      <Link to={to} className="block focus:outline-none focus:ring-2 focus:ring-blue-500 rounded-xl h-full">
        {Inner}
      </Link>
    )
  }
  return Inner
}


