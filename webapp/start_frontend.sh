#!/bin/bash

# AgentInvest P2 Frontend Startup Script
echo "🚀 Starting AgentInvest P2 Frontend..."

# Check if we're in the p2 directory
if [ ! -d "frontend" ]; then
    echo "❌ Error: frontend directory not found. Please run this script from the p2 directory."
    exit 1
fi

# Change to frontend directory
cd frontend

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing frontend dependencies..."
    npm install
    
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies"
        exit 1
    fi
    
    echo "✅ Dependencies installed successfully"
fi

# Update the API endpoint to point to port 8001
echo "🔧 Configuring API endpoint for P2 backend (port 8001)..."

# Check if the API service file exists and update it
if [ -f "src/services/api.js" ]; then
    # Create a backup
    cp src/services/api.js src/services/api.js.backup
    
    # Update the base URL to use port 8001
    sed -i '' 's/localhost:8000/localhost:8001/g' src/services/api.js
    echo "✅ API endpoint updated to use port 8001"
fi

echo "🌐 Starting frontend development server..."
echo "📊 Frontend will be available at: http://localhost:3000"
echo "🔗 Backend API: http://localhost:8001"
echo "🔄 Press Ctrl+C to stop the server"

# Start the development server
npm start
