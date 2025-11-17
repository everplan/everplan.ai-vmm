#!/bin/bash
# Quick start script for AI-VMM with containerized backends

set -e

echo "🚀 Starting AI-VMM with Containerized Backends"
echo "================================================"
echo ""

# Check Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi

# Check Docker Compose
if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "❌ Docker Compose not found. Please install Docker Compose first."
    exit 1
fi

# Check GPU access
if [ ! -d "/dev/dri" ]; then
    echo "⚠️  Warning: /dev/dri not found. Intel GPU may not be available."
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p models config

# Check if models exist
if [ ! -f "models/tinyllama_openvino/openvino_model.xml" ]; then
    echo "⚠️  TinyLlama OpenVINO model not found in models/"
    echo "   Intel backend will need a model to run."
    echo "   You can download models or mount your model directory."
fi

# Pull latest images
echo ""
echo "📥 Pulling Docker images..."
docker pull intelanalytics/ipex-llm-serving-xpu:0.2.0-b2

# Start services
echo ""
echo "🎯 Starting AI-VMM services..."
if docker compose version &> /dev/null; then
    docker compose up -d
else
    docker-compose up -d
fi

# Wait for services to be ready
echo ""
echo "⏳ Waiting for services to start..."
sleep 10

# Check service health
echo ""
echo "🔍 Checking service status..."
if docker compose version &> /dev/null; then
    docker compose ps
else
    docker-compose ps
fi

echo ""
echo "✅ AI-VMM is starting!"
echo ""
echo "📊 Access points:"
echo "   • Web UI:          http://localhost:8000"
echo "   • Core API:        http://localhost:8080"
echo "   • Intel Backend:   http://localhost:8001 (internal)"
echo ""
echo "📝 View logs:"
echo "   docker compose logs -f"
echo ""
echo "🛑 Stop services:"
echo "   docker compose down"
echo ""
echo "🔧 Test Intel GPU detection:"
echo "   docker exec -it \$(docker ps -q -f name=intel-backend) sycl-ls"
echo ""
