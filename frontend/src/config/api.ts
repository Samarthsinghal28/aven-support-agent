// API Configuration
export const API_CONFIG = {
  BASE_URL: process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000",
  WS_URL: process.env.NEXT_PUBLIC_WS_URL || "ws://localhost:8000",
  ENDPOINTS: {
    CHAT: "/chat",
    HEALTH: "/health",
    SESSION: "/sessions",
    WS_VOICE: "/ws/voice"
  }
}

export default API_CONFIG 