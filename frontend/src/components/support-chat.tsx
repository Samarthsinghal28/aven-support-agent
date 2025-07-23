"use client"

import type React from "react"
import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { Avatar } from "@/components/ui/avatar"
import { Card, CardContent, CardFooter, CardHeader } from "@/components/ui/card"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Send, Bot, AlertTriangle } from "lucide-react"
import VapiVoiceCall from "./vapi-voice-call"
import { useVapiService } from "@/hooks/useVapiService"

type Message = {
  id: string
  content: string
  sender: "user" | "agent"
  timestamp: Date
  response_time?: number
}

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"

const initialMessages: Message[] = [
  {
    id: "1",
    content: "Hello! Welcome to Aven support. I'm your AI assistant. How can I help you with our HELOC or credit card products today?",
    sender: "agent",
    timestamp: new Date(),
  },
]

export default function SupportChat() {
  const [messages, setMessages] = useState<Message[]>(initialMessages)
  const [input, setInput] = useState("")
  const [isChatLoading, setIsChatLoading] = useState(false)
  const [sessionId, setSessionId] = useState<string>("")
  const [activeTab, setActiveTab] = useState("chat")
  
  const { 
    isVapiAvailable, 
    assistantId, 
    error: vapiError, 
    isLoading: isVapiLoading 
  } = useVapiService()

  const messagesEndRef = useRef<HTMLDivElement>(null)

  // Generate a unique session ID on component mount
  useEffect(() => {
    const newSessionId = `session_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`
    setSessionId(newSessionId)
  }, [])

  // Scroll to the bottom of the chat when new messages are added
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
  }, [messages])

  // Function to add a new message to the chat history
  const addMessage = (content: string, sender: "user" | "agent", responseTime?: number) => {
    const newMessage: Message = {
      id: Date.now().toString(),
      content,
      sender,
      timestamp: new Date(),
      response_time: responseTime,
    }
    setMessages(prev => [...prev, newMessage])
  }

  // Function to send a text message to the backend
  const sendChatMessage = async (message: string) => {
    if (!message.trim()) return

    setIsChatLoading(true)
    addMessage(message, "user")
    setInput("")

    try {
      const response = await fetch(`${API_BASE_URL}/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "ngrok-skip-browser-warning": "true",
        },
        body: JSON.stringify({ message, session_id: sessionId }),
      })

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`)
      }

      const data = await response.json()
      addMessage(data.response, "agent", data.response_time)
      
    } catch (error) {
      console.error("Error sending message:", error)
      addMessage("Sorry, I'm experiencing technical difficulties. Please try again later.", "agent")
    } finally {
      setIsChatLoading(false)
    }
  }

  // Handle form submission for text chat
  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!input.trim() || isChatLoading) return
    sendChatMessage(input)
  }

  return (
    <Card className="w-full max-w-4xl mx-auto h-[600px] flex flex-col border-0 shadow-lg">
      <CardHeader className="border-b border-gray-100 pb-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <Avatar className="h-10 w-10 bg-blue-100">
              <Bot className="h-6 w-6 text-blue-600" />
            </Avatar>
            <div>
              <h3 className="font-semibold text-lg">Aven Support AI</h3>
              <p className="text-sm text-gray-500">
                {sessionId ? `Session: ${sessionId.slice(-8)}` : "Initializing..."}
              </p>
              {isVapiAvailable && (
                <p className="text-xs text-green-600 font-medium">🎙️ Voice Enabled</p>
              )}
            </div>
          </div>
        </div>
      </CardHeader>

      <CardContent className="flex-1 p-0">
        <Tabs value={activeTab} onValueChange={setActiveTab} className="h-full flex flex-col">
          <TabsList className="grid w-full grid-cols-2 mx-4 mt-4">
            <TabsTrigger value="chat" className="text-sm">💬 Text Chat</TabsTrigger>
            <TabsTrigger value="voice" className="text-sm">🎙️ Voice Chat</TabsTrigger>
          </TabsList>

          <TabsContent value="chat" className="flex-1 flex flex-col mt-4 px-4">
            <ScrollArea className="flex-1 mb-4">
              <div className="space-y-4">
                {messages.map((message) => (
                  <div
                    key={message.id}
                    className={`flex ${message.sender === "user" ? "justify-end" : "justify-start"}`}
                  >
                    <div
                      className={`max-w-[80%] rounded-lg px-4 py-2 ${
                        message.sender === "user"
                          ? "bg-blue-600 text-white"
                          : "bg-gray-100 text-gray-900"
                      }`}
                    >
                      <p className="text-sm">{message.content}</p>
                      <div className="flex items-center justify-between mt-1">
                        <p className="text-xs opacity-70">
                          {message.timestamp.toLocaleTimeString('en-US', { hour12: false })}
                        </p>
                        {message.response_time && (
                          <p className="text-xs opacity-70">
                            {message.response_time.toFixed(2)}s
                          </p>
                        )}
                      </div>
                    </div>
                  </div>
                ))}
                {isChatLoading && (
                  <div className="flex justify-start">
                    <div className="bg-gray-100 rounded-lg px-4 py-2">
                      <div className="flex items-center space-x-2">
                        <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-600"></div>
                        <p className="text-sm text-gray-600">Agent is thinking...</p>
                      </div>
                    </div>
                  </div>
                )}
              </div>
              <div ref={messagesEndRef} />
            </ScrollArea>
          </TabsContent>

          <TabsContent value="voice" className="flex-1 flex flex-col mt-4 px-4">
            <div className="flex-1 flex flex-col items-center justify-center">
              {isVapiLoading && (
                <div className="text-center p-8">
                  <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto mb-4"></div>
                  <h3 className="text-xl font-medium mb-2">Connecting Voice Service...</h3>
                  <p className="text-gray-500">Please wait a moment.</p>
                </div>
              )}
              {vapiError && (
                 <div className="text-center p-8 bg-red-50 rounded-lg">
                  <AlertTriangle className="h-12 w-12 text-red-500 mx-auto mb-4" />
                  <h3 className="text-xl font-medium mb-2 text-red-800">Voice Service Error</h3>
                  <p className="text-red-600">{vapiError}</p>
                  <p className="text-sm text-gray-500 mt-2">Please check the backend connection and refresh.</p>
                </div>
              )}
              {!isVapiLoading && !vapiError && assistantId && (
                <VapiVoiceCall assistantId={assistantId} />
              )}
            </div>
          </TabsContent>
        </Tabs>
      </CardContent>

      <CardFooter className="border-t border-gray-100 pt-4">
        <form onSubmit={handleSubmit} className="flex w-full space-x-2">
          <Input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Type your message..."
            disabled={isChatLoading}
            className="flex-1"
          />
          <Button type="submit" disabled={!input.trim() || isChatLoading} className="px-6">
            {isChatLoading ? (
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white" />
            ) : (
              <Send className="h-4 w-4" />
            )}
          </Button>
        </form>
      </CardFooter>
    </Card>
  )
}
