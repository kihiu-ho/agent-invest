#!/bin/bash

# Runtime environment variable injection for React app
# This script replaces placeholder values in the built React app with actual environment variables

set -e

echo "🔧 Injecting runtime environment variables..."

# Define the build directory
BUILD_DIR="/usr/share/nginx/html"

# Check if build directory exists
if [ ! -d "$BUILD_DIR" ]; then
    echo "❌ Build directory $BUILD_DIR not found!"
    exit 1
fi

# Function to replace environment variables in files
replace_env_vars() {
    local file="$1"
    
    if [ -f "$file" ]; then
        echo "📝 Processing $file..."
        
        # Replace REACT_APP_API_URL
        if [ -n "$REACT_APP_API_URL" ]; then
            sed -i "s|http://localhost:8000|$REACT_APP_API_URL|g" "$file"
            sed -i "s|REACT_APP_API_URL_PLACEHOLDER|$REACT_APP_API_URL|g" "$file"
            echo "   ✅ Updated REACT_APP_API_URL to: $REACT_APP_API_URL"
        fi
        
        # Replace REACT_APP_WS_URL
        if [ -n "$REACT_APP_WS_URL" ]; then
            sed -i "s|ws://localhost:8000|$REACT_APP_WS_URL|g" "$file"
            sed -i "s|REACT_APP_WS_URL_PLACEHOLDER|$REACT_APP_WS_URL|g" "$file"
            echo "   ✅ Updated REACT_APP_WS_URL to: $REACT_APP_WS_URL"
        fi
    fi
}

# Find and process all JavaScript files in the build directory
echo "🔍 Finding JavaScript files to process..."
find "$BUILD_DIR" -name "*.js" -type f | while read -r file; do
    replace_env_vars "$file"
done

# Also process the main HTML file
if [ -f "$BUILD_DIR/index.html" ]; then
    replace_env_vars "$BUILD_DIR/index.html"
fi

echo "✅ Environment variable injection completed!"

# Print current environment variables for debugging
echo "🔍 Current environment variables:"
echo "   REACT_APP_API_URL: ${REACT_APP_API_URL:-'not set'}"
echo "   REACT_APP_WS_URL: ${REACT_APP_WS_URL:-'not set'}"
