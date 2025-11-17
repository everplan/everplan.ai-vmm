# Containerized CPU Backend Plan

## Overview
Move CPU backend (OpenVINO) into a Docker container for consistency with the GPU backend architecture.

## Benefits

1. **Architectural Consistency** - Both backends in containers
2. **Dependency Isolation** - No host conflicts
3. **Version Control** - Lock OpenVINO versions
4. **Portability** - Easy deployment
5. **Clean Solution** - Matches documented containerized strategy

## Implementation

### 1. CPU Backend Container

**Image**: Custom built on `openvino/ubuntu22_runtime:2025.3.0`

**Location**: `docker/cpu-backend/`
- `Dockerfile` - Container definition
- `llm_server.py` - FastAPI server for LLM inference

**Features**:
- OpenAI-compatible API (`/v1/chat/completions`)
- Optimum-Intel for OpenVINO LLM support
- Auto-converts HuggingFace models to OpenVINO IR
- CPU-optimized inference
- Health checks

**Port**: 8002

### 2. Docker Compose Configuration

Updated `docker-compose.yml` with three services:

```yaml
services:
  ai-vmm-core:
    # Main orchestrator
    ports: [8000, 8080]
    
  intel-backend-gpu:
    # IPEX-LLM for Arc B580
    image: intelanalytics/ipex-llm-serving-xpu:0.2.0-b2
    devices: [/dev/dri]
    ports: [8001]
    
  intel-backend-cpu:
    # OpenVINO for CPU
    build: ./docker/cpu-backend
    ports: [8002]
```

### 3. Web API Updates

**File**: `web/vmm_api.py`

**Changes**:
- Added `INTEL_CPU_BACKEND_URL` configuration
- Updated `ChatCompletionRequest` with `device` parameter
- Modified `/api/chat/completions` to route based on device:
  - `device="gpu"` → Port 8001 (IPEX-LLM)
  - `device="cpu"` → Port 8002 (OpenVINO)

### 4. Web Interface

**File**: `web/static/index.html`

**Changes**:
- Device selector shows "CPU (Coming Soon - OpenVINO Container)"
- Removed intrusive warning dialog
- Ready to pass `device` parameter to API

## Build and Deploy

### Quick Start (GPU Only - Currently Running)
```bash
# Already running
docker ps | grep ipex-llm-server
```

### Full Stack (CPU + GPU)
```bash
# Build CPU backend
cd /root/everplan.ai-vmm
docker build -t ai-vmm-cpu-backend ./docker/cpu-backend

# Run CPU backend
docker run -d \
  --name openvino-cpu \
  -p 8002:8002 \
  -e MODEL_NAME="TinyLlama/TinyLlama-1.1B-Chat-v1.0" \
  ai-vmm-cpu-backend

# Or use docker-compose
docker-compose up -d
```

## Testing

### Test CPU Backend
```bash
curl -X POST http://localhost:8002/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "model": "tinyllama",
    "max_tokens": 50
  }'
```

### Test via Web API
```bash
curl -X POST http://localhost:8000/api/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Hello!"}],
    "model": "tinyllama",
    "device": "cpu",
    "max_tokens": 50
  }'
```

## Performance Expectations

### GPU Backend (Current - Working)
- **Throughput**: 80-90 tokens/sec
- **Latency**: 100-300ms
- **Device**: Intel Arc B580
- **Backend**: IPEX-LLM vLLM

### CPU Backend (Planned)
- **Throughput**: 10-20 tokens/sec (estimated)
- **Latency**: 500-2000ms (estimated)
- **Device**: Intel Xeon w7-3455 (24C/48T)
- **Backend**: OpenVINO with optimum-intel

## Architecture Diagram

```
┌─────────────────────────────────────────┐
│         Web Interface (Port 8000)       │
│         Device Selector: CPU/GPU        │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│      FastAPI (vmm_api.py)               │
│      /api/chat/completions              │
│      Routes by device parameter         │
└────────┬──────────────────┬─────────────┘
         │                  │
    device=gpu          device=cpu
         │                  │
         ▼                  ▼
┌──────────────────┐  ┌────────────────────┐
│ IPEX-LLM vLLM    │  │ OpenVINO CPU       │
│ Port 8001        │  │ Port 8002          │
│ Intel Arc B580   │  │ Xeon w7-3455       │
│ 80-90 tok/s      │  │ 10-20 tok/s        │
└──────────────────┘  └────────────────────┘
```

## Next Steps

1. ✅ Created Dockerfile and server script
2. ✅ Updated docker-compose.yml
3. ✅ Modified web API routing
4. ✅ Updated web interface
5. ⏳ Build CPU backend image
6. ⏳ Test CPU inference
7. ⏳ Compare CPU vs GPU performance
8. ⏳ Enable CPU option in web UI

## Files Created/Modified

### New Files
- `docker/cpu-backend/Dockerfile`
- `docker/cpu-backend/llm_server.py`
- `CONTAINERIZED_CPU_BACKEND_PLAN.md` (this file)

### Modified Files
- `docker-compose.yml` - Added CPU backend service
- `web/vmm_api.py` - Added device routing
- `web/static/index.html` - Updated device selector label

## Conclusion

Moving the CPU backend to a container provides:
- ✅ Clean architecture matching GPU backend
- ✅ No host dependency conflicts
- ✅ Easy switching between CPU/GPU
- ✅ Foundation for multi-backend orchestration

Once built and tested, users can seamlessly switch between CPU and GPU inference through the web interface device selector, with both backends running in isolated, reproducible containers.

---
**Status**: Architecture designed, ready to build and test
**Date**: November 17, 2025
