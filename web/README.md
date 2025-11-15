# AI-VMM Web Dashboard

A modern web interface for the AI Virtual Machine Manager, providing real-time hardware monitoring, model inference, and performance benchmarking.

## Features

### 🖥️ Hardware Monitoring
- Real-time detection of CPU, GPU, and NPU devices
- Live system resource usage (CPU, memory)
- Device status and availability

### 🎯 Inference Playground
- Drag-and-drop image upload
- Select model and target device
- Real-time classification results
- Latency measurements

### ⏱️ Performance Benchmarking
- Compare CPU vs GPU performance
- Configurable iteration counts
- Detailed metrics (avg, min, max, throughput)

## Quick Start

```bash
cd /root/everplan.ai-vmm/web
./start.sh
```

The dashboard will be available at:
- **Dashboard**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs
- **Redoc**: http://localhost:8000/redoc

## API Endpoints

### GET /api/hardware
List available hardware devices

### GET /api/models
List available AI models

### POST /api/infer
Run inference on uploaded image
- Form data: `image` (file), `model` (string), `device` (string)

### POST /api/benchmark
Run performance benchmark
- JSON body: `{"device": "all", "iterations": 10}`

### GET /api/stats
Get system statistics

## Architecture

```
┌─────────────────────────────────────────┐
│     Web Dashboard (HTML/JS/CSS)         │
│     http://localhost:8000               │
└──────────────┬──────────────────────────┘
               │ REST API
               ↓
┌─────────────────────────────────────────┐
│     FastAPI Server (Python)             │
│     - HTTP endpoints                    │
│     - Request/response handling         │
└──────────────┬──────────────────────────┘
               │ subprocess calls
               ↓
┌─────────────────────────────────────────┐
│     AI-VMM C++ Binaries                 │
│     - ai_vmm_basic_usage               │
│     - ai_vmm_performance_comparison    │
└─────────────────────────────────────────┘
```

## Development

### Install Dependencies
```bash
cd web
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Run Development Server
```bash
python3 vmm_api.py
```

### Test API Endpoints
```bash
# List hardware
curl http://localhost:8000/api/hardware

# List models
curl http://localhost:8000/api/models

# Run benchmark
curl -X POST http://localhost:8000/api/benchmark \
  -H "Content-Type: application/json" \
  -d '{"device": "all", "iterations": 10}'
```

## Configuration

Edit `vmm_api.py` to configure:
- `VMM_BUILD_DIR`: Path to AI-VMM build directory
- `MODELS_DIR`: Path to ONNX models
- API host/port (default: 0.0.0.0:8000)

## Requirements

- Python 3.8+
- FastAPI 0.104+
- Uvicorn
- AI-VMM compiled binaries in `../build/`

## Future Enhancements

- [ ] WebSocket support for real-time streaming
- [ ] Live webcam inference
- [ ] Model upload and management
- [ ] Multi-user sessions
- [ ] Authentication and API keys
- [ ] Docker containerization
- [ ] Performance charts and visualization
- [ ] Model comparison side-by-side

## Troubleshooting

### Port Already in Use
```bash
# Find process using port 8000
lsof -i :8000
# Kill the process
kill -9 <PID>
```

### Binaries Not Found
Ensure AI-VMM is built:
```bash
cd /root/everplan.ai-vmm
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

### CORS Errors
The API allows all origins by default. For production, configure `allow_origins` in `vmm_api.py`.
