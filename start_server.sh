#!/bin/bash

# 🚀 Aven Support Bot - Server Startup Script
# Starts both backend (FastAPI + Vapi) and frontend (Next.js) servers

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Starting Aven Support Bot with Vapi Integration${NC}"
echo "============================================================"

# Function to check if port is in use
check_port() {
    if lsof -i :$1 >/dev/null 2>&1; then
        echo -e "${YELLOW}⚠️  Port $1 is already in use${NC}"
        return 1
    fi
    return 0
}

# Function to kill background processes on exit
cleanup() {
    echo -e "\n${YELLOW}🧹 Cleaning up processes...${NC}"
    if [ ! -z "$BACKEND_PID" ]; then
        kill $BACKEND_PID 2>/dev/null || true
    fi
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null || true
    fi
    echo -e "${GREEN}✅ Cleanup complete${NC}"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Check required ports
echo -e "${BLUE}🔍 Checking ports...${NC}"
if ! check_port 8000; then
    echo -e "${RED}❌ Backend port 8000 is in use. Please free it and try again.${NC}"
    exit 1
fi

if ! check_port 3000; then
    echo -e "${RED}❌ Frontend port 3000 is in use. Please free it and try again.${NC}"
    exit 1
fi

# Check backend environment
echo -e "${BLUE}🔧 Checking backend environment...${NC}"
if [ ! -f "aven-support-backend/.env" ]; then
    echo -e "${YELLOW}⚠️  aven-support-backend/.env not found. Creating from template...${NC}"
    cp aven-support-backend/env.example aven-support-backend/.env
    echo -e "${RED}❌ Please configure your API keys in aven-support-backend/.env and run again${NC}"
    exit 1
fi

# Check if virtual environment exists
if [ ! -d "aven-support-backend/venv" ]; then
    echo -e "${YELLOW}⚠️  Virtual environment not found. Creating...${NC}"
    cd aven-support-backend
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    cd ..
fi

# Check frontend dependencies
echo -e "${BLUE}📦 Checking frontend dependencies...${NC}"
if [ ! -d "frontend/node_modules" ]; then
    echo -e "${YELLOW}⚠️  Frontend dependencies not installed. Installing...${NC}"
    cd frontend
    npm install
    cd ..
fi

# Check frontend environment
if [ ! -f "frontend/.env.local" ]; then
    echo -e "${YELLOW}⚠️  frontend/.env.local not found. Creating...${NC}"
    echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > frontend/.env.local
fi

# Start backend server
echo -e "${GREEN}🔌 Starting backend server (FastAPI + Vapi)...${NC}"
cd aven-support-backend
source venv/bin/activate

# Check API keys
echo -e "${BLUE}🔑 Checking API configuration...${NC}"
python -c "
import os
from dotenv import load_dotenv
load_dotenv()

print('Required keys:')
required = ['OPENAI_API_KEY', 'PINECONE_API_KEY']
optional = ['SERPER_API_KEY', 'VAPI_API_KEY']

all_good = True
for key in required:
    if os.getenv(key):
        print(f'  ✅ {key}')
    else:
        print(f'  ❌ {key} (REQUIRED)')
        all_good = False

print('Optional keys:')
for key in optional:
    if os.getenv(key):
        print(f'  ✅ {key}')
    else:
        print(f'  ⚠️  {key} (optional)')

if not all_good:
    print('\\n❌ Missing required API keys. Please check aven-support-backend/.env')
    exit(1)

print('\\n🎙️ Vapi Integration Status:')
if os.getenv('VAPI_API_KEY'):
    print('  ✅ Voice calls available')
else:
    print('  ⚠️  Voice calls disabled (add VAPI_API_KEY to enable)')
"

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Backend configuration check failed${NC}"
    exit 1
fi

# Start backend in background
echo -e "${GREEN}▶️  Starting FastAPI server on http://localhost:8000${NC}"
python server.py > ../aven-support-backend.log 2>&1 &
BACKEND_PID=$!

cd ..

# Wait for backend to start
echo -e "${BLUE}⏳ Waiting for backend to start...${NC}"
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null; then
        echo -e "${GREEN}✅ Backend is ready!${NC}"
        break
    fi
    sleep 1
    if [ $i -eq 30 ]; then
        echo -e "${RED}❌ Backend failed to start${NC}"
        echo -e "${YELLOW}Check aven-support-backend.log for details${NC}"
        exit 1
    fi
done

# Show backend health
echo -e "${BLUE}🏥 Backend Health Check:${NC}"
curl -s http://localhost:8000/health | python -m json.tool

# --- Frontend ---
echo -e "${BLUE}--- Starting Frontend ---${NC}"
cd frontend

# Build the Next.js application
echo -e "${BLUE}Building frontend application...${NC}"
npm run build > ../frontend.log 2>&1

if [ $? -ne 0 ]; then
    echo -e "${RED}❌ Frontend build failed. Check frontend.log for details.${NC}"
    exit 1
fi
echo -e "${GREEN}✅ Frontend build successful.${NC}"

# Start the Next.js application in the background
echo -e "${BLUE}Starting frontend server...${NC}"
npm run start >> ../frontend.log 2>&1 &
FRONTEND_PID=$!
cd ..

# Wait for frontend to be ready
echo -e "${BLUE}⏳ Waiting for frontend to start...${NC}"
for i in {1..60}; do
    if curl -s http://localhost:3000 > /dev/null; then
        echo -e "${GREEN}✅ Frontend is ready!${NC}"
        break
    fi
    sleep 1
    if [ $i -eq 60 ]; then
        echo -e "${RED}❌ Frontend failed to start. Check frontend.log for details.${NC}"
        kill $BACKEND_PID
        exit 1
    fi
done

# Show startup success
echo ""
echo -e "${GREEN}🎉 AVEN SUPPORT BOT READY!${NC}"
echo "============================================================"
echo -e "${BLUE}📱 Frontend:${NC}     http://localhost:3000"
echo -e "${BLUE}🔌 Backend API:${NC}  http://localhost:8000"
echo -e "${BLUE}📚 API Docs:${NC}     http://localhost:8000/docs"
echo ""
echo -e "${GREEN}✨ Features Available:${NC}"
echo "  💬 Text Chat"
echo "  🎙️ Voice Chat (WebSocket)"
if grep -q "VAPI_API_KEY=" aven-support-backend/.env && [ "$(grep VAPI_API_KEY= aven-support-backend/.env | cut -d'=' -f2)" != "your_vapi_key_here" ]; then
    echo "  🎤 Vapi Voice Calls (Real STT/TTS)"
else
    echo "  ⚠️  Vapi Voice Calls (disabled - add VAPI_API_KEY)"
fi
echo ""
echo -e "${YELLOW}📝 Logs:${NC}"
echo "  Backend:  tail -f aven-support-backend.log"
echo "  Frontend: tail -f frontend.log"
echo ""
echo -e "${BLUE}Press Ctrl+C to stop all servers${NC}"

# Keep script running and show live logs
while true; do
    sleep 1
done 