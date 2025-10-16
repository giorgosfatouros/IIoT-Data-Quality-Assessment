import Card from '../components/Card'
import Footer from '../components/Footer'
import { Database, BarChart2, AlertTriangle, Activity, CheckCircle, Bot } from 'lucide-react'

export default function Home() {
  return (
    <div className="h-full flex flex-col space-y-8">
      <section>
        <h1 className="text-3xl font-bold mb-4 text-white">Welcome</h1>
        <p className="text-gray-300 text-lg leading-relaxed max-w-4xl">
          Analyze and assess the quality of IIoT sensor data with LeanXcale-powered aggregation and fast analytics.
        </p>
      </section>

      <section className="flex-1">
        <h2 className="text-xl font-semibold mb-6 text-white">Get Started</h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 2xl:grid-cols-4 gap-6 auto-rows-fr">
          <Card title="Data Loading" description="Connect to LeanXcale and select a machine table." icon={<Database size={24} />} to="/data-loading" />
          <Card title="Data Visualization" description="Explore trends, distributions and correlations." icon={<BarChart2 size={24} />} to="/data-visualization" />
          <Card title="Missing Values" description="Identify gaps and quantify missing readings." icon={<AlertTriangle size={24} />} to="/missing-values" />
          <Card title="Invalid Values" description="Spot invalid readings and alarms." icon={<Activity size={24} />} to="/invalid-values" />
          <Card title="Data Quality" description="Assess completeness, accuracy and consistency." icon={<CheckCircle size={24} />} to="/data-quality" />
          <Card title="DQA Agent" description="Chat with AI agent about your sensor data quality." icon={<Bot size={24} />} to="/dqa-agent" />
        </div>
      </section>
      
      <Footer />
    </div>
  )
}


