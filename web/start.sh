#!/bin/bash
# AI-VMM Web Server Startup Script

set -e

echo "🚀 AI-VMM Web Dashboard"
echo "========================"
echo ""

# Check if we're in the right directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check for virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating Python virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing Python dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Check if build directory exists
BUILD_DIR="../build"
if [ ! -d "$BUILD_DIR" ]; then
    echo "❌ Error: Build directory not found at $BUILD_DIR"
    echo "Please build the project first:"
    echo "  cd /root/everplan.ai-vmm && mkdir -p build && cd build && cmake .. && make"
    exit 1
fi

# Check if binaries exist
BASIC_USAGE="$BUILD_DIR/examples/basic_usage/ai_vmm_basic_usage"
PERF_BENCH="$BUILD_DIR/examples/performance_comparison/ai_vmm_performance_comparison"

if [ ! -f "$BASIC_USAGE" ]; then
    echo "⚠️  Warning: basic_usage binary not found at $BASIC_USAGE"
fi

if [ ! -f "$PERF_BENCH" ]; then
    echo "⚠️  Warning: performance_comparison binary not found at $PERF_BENCH"
fi

echo ""
echo "✅ Environment ready!"
echo ""
echo "🌐 Starting web server..."
echo "   Dashboard: http://localhost:8000"
echo "   API Docs:  http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Start the server
python3 vmm_api.py
