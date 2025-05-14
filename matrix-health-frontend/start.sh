#!/bin/bash

# Start the frontend
echo "Starting Matrix Health Terminal frontend..."
npm run dev &
FRONTEND_PID=$!

echo "Frontend started on http://localhost:3000"

# Check if FastAPI backend is running
if pgrep -f "uvicorn Bot_Integration_with_MCP_Tools:app" > /dev/null; then
    echo "FastAPI backend is already running."
else
    echo "Starting FastAPI backend..."
    cd ../
    source .venv/bin/activate
    uvicorn Bot_Integration_with_MCP_Tools:app --reload &
    BACKEND_PID=$!
    echo "Backend started on http://localhost:8000"
    cd matrix-health-frontend
fi

echo "Matrix Health Terminal is now running!"
echo "Frontend: http://localhost:3000"
echo "Backend: http://localhost:8000"

# Wait for Ctrl+C
echo "Press Ctrl+C to stop both servers..."
trap "kill $FRONTEND_PID $BACKEND_PID 2>/dev/null" EXIT
wait 