"use client"

import type React from "react"

import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Card } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { Skeleton } from "@/components/ui/skeleton"
import { Send, Upload, Bot, User, Database } from "lucide-react"
import { cn } from "@/lib/utils"
import ReactMarkdown from "react-markdown"
import { chatAPI, datasetAPI } from "@/lib/api"

interface Message {
  id: string
  role: "user" | "assistant"
  content: string
  timestamp: Date
  isLoading?: boolean
  execution_time?: number
}

interface ChatInterfaceProps {
  selectedDataset: string | null
}

export function ChatInterface({ selectedDataset }: ChatInterfaceProps) {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "1",
      role: "assistant",
      content:
        "Hello! I'm your AI business insights assistant. Upload a dataset or select one from your datasets to start analyzing your data. I can help you discover trends, generate reports, and answer questions about your business data.",
      timestamp: new Date(),
    },
  ])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  useEffect(() => {
    // Load chat history when dataset changes
    if (selectedDataset) {
      loadChatHistory(parseInt(selectedDataset))
    }
  }, [selectedDataset])

  const loadChatHistory = async (datasetId: number) => {
    try {
      const history = await chatAPI.getChatHistory(datasetId)
      const historyMessages: Message[] = history.map((query: any) => [
        {
          id: `${query.id}-user`,
          role: "user" as const,
          content: query.question,
          timestamp: new Date(query.created_at),
        },
        {
          id: `${query.id}-assistant`,
          role: "assistant" as const,
          content: query.answer || "No response available",
          timestamp: new Date(query.created_at),
          execution_time: query.execution_time,
        },
      ]).flat()

      setMessages(prev => [
        prev[0], // Keep the initial welcome message
        ...historyMessages,
      ])
    } catch (error) {
      console.error('Error loading chat history:', error)
    }
  }

  const handleSend = async () => {
    if (!input.trim()) return

    const userMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: input,
      timestamp: new Date(),
    }

    setMessages((prev) => [...prev, userMessage])
    setInput("")
    setIsLoading(true)

    try {
      const response = await chatAPI.sendMessage(
        input,
        selectedDataset ? parseInt(selectedDataset) : undefined
      )

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: response.response,
        timestamp: new Date(),
        execution_time: response.execution_time,
      }

      setMessages((prev) => [...prev, assistantMessage])
    } catch (error) {
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: "I apologize, but I encountered an error while processing your request. Please try again.",
        timestamp: new Date(),
      }

      setMessages((prev) => [...prev, errorMessage])
    } finally {
      setIsLoading(false)
    }
  }

  const handleFileUpload = () => {
    fileInputRef.current?.click()
  }

  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return

    const uploadMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: `📁 Uploading dataset: ${file.name}`,
      timestamp: new Date(),
    }

    setMessages((prev) => [...prev, uploadMessage])

    try {
      const result = await datasetAPI.uploadDataset(file, file.name)
      
      const confirmMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: `Perfect! I've successfully processed your dataset "${file.name}". 

**Dataset Summary:**
- **File Size:** ${(result.file_info.size / 1024 / 1024).toFixed(2)} MB
- **Rows:** ${result.file_info.rows.toLocaleString()}
- **Columns:** ${result.file_info.columns}

Your dataset is now ready for analysis! Try asking me questions like:
- "What are the key trends in this data?"
- "Show me a summary of the data"
- "Are there any anomalies I should know about?"

What would you like to explore first?`,
        timestamp: new Date(),
      }

      setMessages((prev) => [...prev, confirmMessage])
    } catch (error) {
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: `Sorry, I couldn't process the file "${file.name}". Please make sure it's a valid CSV, Excel, or JSON file and try again.`,
        timestamp: new Date(),
      }

      setMessages((prev) => [...prev, errorMessage])
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="flex flex-col h-full bg-gray-950">
      {/* Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-800 bg-gray-900/50">
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-teal-500 rounded-lg flex items-center justify-center">
            <Bot className="w-5 h-5 text-white" />
          </div>
          <div>
            <h2 className="text-lg font-semibold text-white">AI Business Assistant</h2>
            {selectedDataset && (
              <div className="flex items-center space-x-2">
                <Database className="w-3 h-3 text-teal-400" />
                <span className="text-xs text-teal-400">Connected to: {selectedDataset}</span>
              </div>
            )}
          </div>
        </div>
        <div className="flex items-center space-x-2">
          <Badge variant="secondary" className="bg-green-600/20 text-green-400 border border-green-600/30">
            <div className="w-2 h-2 bg-green-400 rounded-full mr-2" />
            Online
          </Badge>
        </div>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {messages.map((message) => (
          <div
            key={message.id}
            className={cn("flex items-start space-x-3", message.role === "user" ? "justify-end" : "justify-start")}
          >
            {message.role === "assistant" && (
              <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-teal-500 rounded-lg flex items-center justify-center flex-shrink-0">
                <Bot className="w-4 h-4 text-white" />
              </div>
            )}

            <Card
              className={cn(
                "max-w-[80%] p-4",
                message.role === "user"
                  ? "bg-blue-600 text-white border-blue-600"
                  : "bg-gray-800 text-gray-100 border-gray-700",
              )}
            >
              <div className="prose prose-sm max-w-none">
                {message.role === "assistant" ? (
                  <ReactMarkdown
                    components={{
                      p: ({ children }) => <p className="text-gray-100 mb-2 last:mb-0">{children}</p>,
                      strong: ({ children }) => <strong className="text-white font-semibold">{children}</strong>,
                      ul: ({ children }) => <ul className="text-gray-100 ml-4 mb-2">{children}</ul>,
                      ol: ({ children }) => <ol className="text-gray-100 ml-4 mb-2">{children}</ol>,
                      li: ({ children }) => <li className="mb-1">{children}</li>,
                      table: ({ children }) => (
                        <table className="w-full border-collapse border border-gray-600 my-2">{children}</table>
                      ),
                      th: ({ children }) => (
                        <th className="border border-gray-600 px-2 py-1 bg-gray-700 text-white text-left">
                          {children}
                        </th>
                      ),
                      td: ({ children }) => (
                        <td className="border border-gray-600 px-2 py-1 text-gray-100">{children}</td>
                      ),
                    }}
                  >
                    {message.content}
                  </ReactMarkdown>
                ) : (
                  <p className="text-white">{message.content}</p>
                )}
              </div>
              <div className="flex items-center justify-between mt-2 pt-2 border-t border-gray-600/30">
                <span className="text-xs text-gray-400">{message.timestamp.toLocaleTimeString()}</span>
              </div>
            </Card>

            {message.role === "user" && (
              <div className="w-8 h-8 bg-gradient-to-br from-purple-500 to-pink-500 rounded-lg flex items-center justify-center flex-shrink-0">
                <User className="w-4 h-4 text-white" />
              </div>
            )}
          </div>
        ))}

        {isLoading && (
          <div className="flex items-start space-x-3">
            <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-teal-500 rounded-lg flex items-center justify-center flex-shrink-0">
              <Bot className="w-4 h-4 text-white" />
            </div>
            <Card className="max-w-[80%] p-4 bg-gray-800 border-gray-700">
              <div className="space-y-2">
                <Skeleton className="h-4 w-full bg-gray-700" />
                <Skeleton className="h-4 w-3/4 bg-gray-700" />
                <Skeleton className="h-4 w-1/2 bg-gray-700" />
              </div>
            </Card>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {/* Input */}
      <div className="p-4 border-t border-gray-800 bg-gray-900/50">
        <div className="flex items-center space-x-2">
          <Button
            variant="outline"
            size="sm"
            onClick={handleFileUpload}
            className="border-gray-600 text-gray-300 hover:bg-gray-800 bg-transparent"
          >
            <Upload className="w-4 h-4" />
          </Button>
          <div className="flex-1 relative">
            <Input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask me about your business data..."
              className="bg-gray-800 border-gray-600 text-white placeholder-gray-400 pr-12"
              disabled={isLoading}
            />
            <Button
              onClick={handleSend}
              disabled={!input.trim() || isLoading}
              size="sm"
              className="absolute right-1 top-1 bg-blue-600 hover:bg-blue-700"
            >
              <Send className="w-4 h-4" />
            </Button>
          </div>
        </div>
        <input
          ref={fileInputRef}
          type="file"
          accept=".csv,.xlsx,.json"
          onChange={handleFileChange}
          className="hidden"
        />
      </div>
    </div>
  )
}
