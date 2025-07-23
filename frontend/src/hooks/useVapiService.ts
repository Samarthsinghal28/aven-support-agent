"use client"

import { useState, useEffect } from "react"

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000"
console.log(`[useVapiService] API_BASE_URL: ${API_BASE_URL}`)

interface VapiServiceState {
  isVapiAvailable: boolean
  assistantId: string | null
  error: string | null
  isLoading: boolean
}

export const useVapiService = (): VapiServiceState => {
  const [state, setState] = useState<VapiServiceState>({
    isVapiAvailable: false,
    assistantId: null,
    error: null,
    isLoading: true,
  })

  useEffect(() => {
    let isMounted = true

    const initializeVapi = async () => {
      try {
        // Step 1: Check backend health and Vapi availability
        setState(prev => ({ ...prev, isLoading: true, error: null }))
        console.log("Checking backend health...")
        const healthResponse = await fetch(`${API_BASE_URL}/health`, {
          headers: { "ngrok-skip-browser-warning": "true" },
        })

        if (!healthResponse.ok) {
          throw new Error(`Health check failed: ${healthResponse.status}`)
        }

        const healthData = await healthResponse.json()
        if (!isMounted) return

        if (!healthData.vapi_available) {
          throw new Error("Vapi service is not available on the backend.")
        }
        
        setState(prev => ({ ...prev, isVapiAvailable: true }))
        console.log("Vapi is available. Creating assistant...")

        // Step 2: Create the Vapi assistant
        const assistantResponse = await fetch(`${API_BASE_URL}/vapi/assistant`, {
          method: "POST",
          headers: { "ngrok-skip-browser-warning": "true" },
        })

        if (!assistantResponse.ok) {
          throw new Error(`Assistant creation failed: ${assistantResponse.status}`)
        }

        const assistantData = await assistantResponse.json()
        if (!isMounted) return
        
        if (!assistantData.assistant_id) {
          throw new Error("Assistant ID not found in the response.")
        }

        console.log(`Assistant created successfully: ${assistantData.assistant_id}`)
        setState(prev => ({
          ...prev,
          assistantId: assistantData.assistant_id,
          isLoading: false,
        }))

      } catch (err) {
        if (isMounted) {
          const errorMessage = err instanceof Error ? err.message : "An unknown error occurred."
          console.error("Error initializing Vapi service:", errorMessage)
          setState(prev => ({ ...prev, error: errorMessage, isLoading: false }))
        }
      }
    }

    initializeVapi()

    return () => {
      isMounted = false
    }
  }, []) // Empty dependency array ensures this runs once on mount

  return state
} 