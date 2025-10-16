import { useState, useEffect, useRef } from 'react'
import { Send, Bot, User, Loader2, AlertCircle, Database, MessageSquare } from 'lucide-react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import Footer from '../components/Footer'

interface ChatMessage {
  role: 'user' | 'assistant'
  content: string
}

export default function DQAAgent() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [isLoading, setIsLoading] = useState(false)
  const [isStreaming, setIsStreaming] = useState(false)
  const [thinkingStatus, setThinkingStatus] = useState<string>('')
  const [toolCalls, setToolCalls] = useState<string[]>([])
  const [error, setError] = useState<string | null>(null)
  const [tables, setTables] = useState<string[]>([])
  const [selectedTable, setSelectedTable] = useState<string>('')
  const messagesEndRef = useRef<HTMLDivElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  // Fetch available tables on mount
  useEffect(() => {
    const fetchTables = async () => {
      try {
        const response = await fetch('http://localhost:8000/tables')
        if (response.ok) {
          const data = await response.json()
          setTables(data)
          if (data.length > 0) {
            setSelectedTable(data[0])
          }
        }
      } catch (err) {
        console.error('Error fetching tables:', err)
      }
    }
    fetchTables()
  }, [])

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || isLoading) return

    const userMessage = input.trim()
    setInput('')
    setError(null)

    // Add user message to chat
    const newMessages = [...messages, { role: 'user' as const, content: userMessage }]
    setMessages(newMessages)
    setIsLoading(true)

    // Add placeholder for streaming assistant message
    const assistantMessageIndex = newMessages.length
    setMessages([...newMessages, { role: 'assistant' as const, content: '' }])
    setIsStreaming(true)
    setToolCalls([])
    setThinkingStatus('')

    try {
      const response = await fetch('http://localhost:8000/agent/chat/stream', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          message: userMessage,
          conversation_history: messages,
          table_name: selectedTable || null,
        }),
      })

      if (!response.ok) {
        const errorData = await response.json()
        throw new Error(errorData.detail || 'Failed to get response from agent')
      }

      // Handle streaming response
      const reader = response.body?.getReader()
      const decoder = new TextDecoder()
      let accumulatedContent = ''

      if (reader) {
        while (true) {
          const { done, value } = await reader.read()
          if (done) break

          const chunk = decoder.decode(value, { stream: true })
          const lines = chunk.split('\n')

          for (const line of lines) {
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.substring(6))
                
                if (data.type === 'thinking') {
                  setThinkingStatus(data.content)
                } else if (data.type === 'tool') {
                  setToolCalls(prev => [...prev, data.content])
                } else if (data.type === 'content') {
                  accumulatedContent += data.content
                  setThinkingStatus('')  // Clear thinking status when content starts
                  // Update the assistant message with accumulated content
                  setMessages(prev => {
                    const updated = [...prev]
                    updated[assistantMessageIndex] = {
                      role: 'assistant',
                      content: accumulatedContent
                    }
                    return updated
                  })
                } else if (data.type === 'done') {
                  // Final update with complete message
                  accumulatedContent = data.content
                  setMessages(prev => {
                    const updated = [...prev]
                    updated[assistantMessageIndex] = {
                      role: 'assistant',
                      content: accumulatedContent
                    }
                    return updated
                  })
                  setIsStreaming(false)
                  setThinkingStatus('')
                  setToolCalls([])
                } else if (data.type === 'error') {
                  throw new Error(data.content)
                }
              } catch (parseError) {
                console.error('Error parsing SSE data:', parseError)
              }
            }
          }
        }
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'An error occurred')
      // Remove the placeholder assistant message if there was an error
      setMessages(newMessages)
      setIsStreaming(false)
      setThinkingStatus('')
      setToolCalls([])
    } finally {
      setIsLoading(false)
    }
  }

  const handleSuggestionClick = (suggestion: string) => {
    setInput(suggestion)
  }

  const suggestions = [
    'Generate a complete data quality report for all sensors',
    'Which sensors have alarm violations? How many?',
    'Show me data completeness - any missing readings?',
    'What is sensor 33VI603 and what are its alarm thresholds?',
    'Analyze pressure sensors for threshold violations',
    'Which sensors have the highest alarm rates?',
  ]

  return (
    <div className="h-full flex flex-col space-y-4">
      {/* Header */}
      <div className="flex-shrink-0">
        <div className="flex items-center gap-3 mb-2">
          <div className="p-3 bg-gradient-to-r from-blue-600 to-indigo-600 rounded-lg shadow-lg">
            <MessageSquare className="text-white" size={32} />
          </div>
          <div className="flex-1">
            <div className="flex items-center justify-between">
              <h1 className="text-3xl font-bold text-white">DQA Agent</h1>
              {tables.length > 0 && (
                <div className="flex items-center gap-2">
                  <Database size={18} className="text-gray-400" />
                  <select
                    value={selectedTable}
                    onChange={(e) => setSelectedTable(e.target.value)}
                    className="bg-gray-700 text-white px-3 py-1.5 rounded-md border border-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500 text-sm"
                  >
                    {tables.map((table) => (
                      <option key={table} value={table}>
                        {table}
                      </option>
                    ))}
                  </select>
                </div>
              )}
            </div>
            <p className="text-gray-300 text-sm mt-1">
              Ask questions about machine sensor data, alarm violations, missing readings, and thresholds. The agent analyzes hourly aggregated data from your equipment.
            </p>
          </div>
        </div>
      </div>

      {/* Chat Container */}
      <div className="flex-1 bg-gray-800/50 rounded-lg border border-gray-700 flex flex-col overflow-hidden">
        {/* Messages */}
        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-gray-400 space-y-6">
              <div className="text-center space-y-2">
                <Bot size={48} className="mx-auto text-blue-500" />
                <h3 className="text-lg font-semibold text-gray-300">Welcome to DQA Agent</h3>
                <p className="text-sm max-w-md">
                  I analyze hourly aggregated machine sensor data (collected every 10 seconds). I can identify alarm violations, missing readings, explain thresholds, and provide insights about equipment health.
                </p>
              </div>
              
              {/* Suggestion chips */}
              <div className="space-y-2 w-full max-w-2xl">
                <p className="text-xs text-gray-500 text-center">Try asking:</p>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                  {suggestions.map((suggestion, idx) => (
                    <button
                      key={idx}
                      onClick={() => handleSuggestionClick(suggestion)}
                      className="px-4 py-2 bg-gray-700/50 hover:bg-gray-700 border border-gray-600 rounded-lg text-sm text-left transition-colors"
                    >
                      {suggestion}
                    </button>
                  ))}
                </div>
              </div>
            </div>
          ) : (
            <>
              {messages.map((message, idx) => (
                <div key={idx}>
                  <div
                    className={`flex gap-3 ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}
                  >
                    {message.role === 'assistant' && (
                      <div className="flex-shrink-0 w-8 h-8 rounded-full bg-blue-600 flex items-center justify-center">
                        <Bot size={18} className="text-white" />
                      </div>
                    )}
                    
                    <div
                      className={`max-w-[80%] rounded-lg px-4 py-2.5 ${
                        message.role === 'user'
                          ? 'bg-blue-600 text-white'
                          : 'bg-gray-700 text-gray-100'
                      }`}
                    >
                      {message.role === 'assistant' ? (
                        <div className="text-sm prose prose-invert prose-sm max-w-none">
                          <ReactMarkdown
                            remarkPlugins={[remarkGfm]}
                            components={{
                              h1: ({ node, ...props }) => <h1 className="text-xl font-bold mt-4 mb-2 text-white" {...props} />,
                              h2: ({ node, ...props }) => <h2 className="text-lg font-bold mt-3 mb-2 text-white" {...props} />,
                              h3: ({ node, ...props }) => <h3 className="text-base font-bold mt-2 mb-1 text-white" {...props} />,
                              p: ({ node, ...props }) => <p className="mb-2 text-gray-100" {...props} />,
                              ul: ({ node, ...props }) => <ul className="list-disc list-inside mb-2 space-y-1" {...props} />,
                              ol: ({ node, ...props }) => <ol className="list-decimal list-inside mb-2 space-y-1" {...props} />,
                              li: ({ node, ...props }) => <li className="text-gray-100" {...props} />,
                              table: ({ node, ...props }) => (
                                <div className="overflow-x-auto my-3">
                                  <table className="min-w-full border-collapse border border-gray-600 text-xs" {...props} />
                                </div>
                              ),
                              thead: ({ node, ...props }) => <thead className="bg-gray-800" {...props} />,
                              tbody: ({ node, ...props }) => <tbody {...props} />,
                              tr: ({ node, ...props }) => <tr className="border-b border-gray-600" {...props} />,
                              th: ({ node, ...props }) => <th className="border border-gray-600 px-3 py-2 text-left font-semibold text-white" {...props} />,
                              td: ({ node, ...props }) => <td className="border border-gray-600 px-3 py-2 text-gray-100" {...props} />,
                              code: ({ node, inline, ...props }: any) => 
                                inline ? (
                                  <code className="bg-gray-800 px-1.5 py-0.5 rounded text-blue-300 font-mono text-xs" {...props} />
                                ) : (
                                  <code className="block bg-gray-800 p-3 rounded my-2 overflow-x-auto font-mono text-xs text-gray-100" {...props} />
                                ),
                              pre: ({ node, ...props }) => <pre className="bg-gray-800 rounded my-2 overflow-x-auto" {...props} />,
                              blockquote: ({ node, ...props }) => <blockquote className="border-l-4 border-blue-500 pl-4 italic text-gray-300 my-2" {...props} />,
                              strong: ({ node, ...props }) => <strong className="font-bold text-white" {...props} />,
                              em: ({ node, ...props }) => <em className="italic text-gray-200" {...props} />,
                              hr: ({ node, ...props }) => <hr className="border-gray-600 my-3" {...props} />,
                              a: ({ node, ...props }) => <a className="text-blue-400 hover:text-blue-300 underline" {...props} />,
                            }}
                          >
                            {message.content}
                          </ReactMarkdown>
                          {isStreaming && idx === messages.length - 1 && (
                            <span className="inline-block w-2 h-4 ml-1 bg-blue-400 animate-pulse"></span>
                          )}
                        </div>
                      ) : (
                        <div className="text-sm whitespace-pre-wrap break-words">
                          {message.content}
                        </div>
                      )}
                    </div>
                    
                    {message.role === 'user' && (
                      <div className="flex-shrink-0 w-8 h-8 rounded-full bg-gray-600 flex items-center justify-center">
                        <User size={18} className="text-white" />
                      </div>
                    )}
                  </div>
                  
                  {/* Show tool calls and thinking status only for the last message when streaming */}
                  {message.role === 'assistant' && idx === messages.length - 1 && isStreaming && (
                    <div className="flex gap-3 justify-start mt-2 ml-11">
                      <div className="flex flex-col gap-1">
                        {thinkingStatus && (
                          <div className="text-xs text-gray-400 italic flex items-center gap-2">
                            <Loader2 size={14} className="animate-spin" />
                            {thinkingStatus}
                          </div>
                        )}
                        {toolCalls.length > 0 && (
                          <div className="flex flex-wrap gap-1">
                            {toolCalls.map((tool, toolIdx) => (
                              <div
                                key={toolIdx}
                                className="text-xs px-2 py-1 bg-gray-800/50 border border-gray-600 rounded text-gray-300"
                              >
                                {tool}
                              </div>
                            ))}
                          </div>
                        )}
                      </div>
                    </div>
                  )}
                </div>
              ))}
              
              <div ref={messagesEndRef} />
            </>
          )}
        </div>

        {/* Error Display */}
        {error && (
          <div className="px-4 py-2 bg-red-900/30 border-t border-red-800/50 flex items-center gap-2 text-red-300 text-sm">
            <AlertCircle size={16} />
            <span>{error}</span>
          </div>
        )}

        {/* Input Form */}
        <div className="border-t border-gray-700 p-4">
          <form onSubmit={handleSendMessage} className="flex gap-2">
            <input
              type="text"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Ask about your sensor data quality..."
              disabled={isLoading}
              className="flex-1 bg-gray-700 text-white px-4 py-2.5 rounded-lg border border-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:opacity-50 disabled:cursor-not-allowed placeholder:text-gray-400"
            />
            <button
              type="submit"
              disabled={isLoading || !input.trim()}
              className="px-4 py-2.5 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white rounded-lg font-medium transition-colors flex items-center gap-2"
            >
              {isLoading ? (
                <Loader2 size={18} className="animate-spin" />
              ) : (
                <Send size={18} />
              )}
              <span className="hidden sm:inline">Send</span>
            </button>
          </form>
        </div>
      </div>

      <Footer />
    </div>
  )
}

