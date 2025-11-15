# AI-VMM Development Roadmap & Configuration

## 🎯 Project Vision

AI-VMM is a universal AI accelerator abstraction layer - think **"Kubernetes for AI accelerators"**. Unlike Ollama (LLM-focused runtime), AI-VMM orchestrates ANY AI model type across heterogeneous hardware (Intel CPU/GPU/NPU, NVIDIA, AMD).

## 📋 Development Roadmap

### ✅ Phase 1: Foundation (COMPLETED)
- [x] Intel backend with CPU/GPU/NPU discovery
- [x] ONNX model loading via OpenVINO
- [x] Hardware constraint-based deployment
- [x] Performance benchmarking tool (CPU vs GPU)
- [x] Multi-device enumeration
- [x] MobileNetV2 inference working

### ✅ Phase 2: Service Architecture (COMPLETED)
**Goal**: Transform VMM from library into a persistent daemon/service

- [x] **Web UI + REST API** ✅ COMPLETED
  - [x] FastAPI backend wrapping VMM C++ library
  - [x] Modern web dashboard with real-time metrics
  - [x] Live inference playground (image upload → classification & detection)
  - [x] Hardware monitoring and visualization (2x Battlemage GPUs detected)
  - [x] Performance benchmarking interface
  - [x] System statistics (CPU, memory)
  - [x] VMM runtime diagnostics tab (backends, providers, CPU features)
  - [x] Example image selector (bus.jpg, zidane.jpg)
  - [x] Persistent inference server (10x faster than subprocess)
  - [ ] WebSocket support for streaming results (future)

**Web Dashboard Running**: http://localhost:8000  
**API Documentation**: http://localhost:8000/docs  
**Performance**: GPU 28ms vs CPU 45ms (1.6x speedup on YOLOv8n)

### 🚧 Phase 3: Model Zoo Expansion (IN PROGRESS - Week 3-4)
**Goal**: Prove versatility across model types

**Priority Models** (ONNX format):
- [x] **MobileNetV2** (image classification) - 14MB, CPU/GPU working
- [x] **YOLOv8n** (object detection) ✅ **COMPLETED**
  - [x] ONNX model integration (13MB)
  - [x] Custom preprocessing (640x640 letterbox, normalization)
  - [x] Non-maximum suppression (NMS) post-processing
  - [x] COCO 80-class detection
  - [x] Bounding box visualization on web UI canvas
  - [x] Persistent server for production performance
  - [x] Example images (bus.jpg, zidane.jpg) with 4-5 detections each
  - [x] GPU acceleration proven (28ms vs 45ms CPU)
- [ ] **YOLOv8s/m** (larger variants) - for better GPU speedup demo
- [ ] TinyLlama-1.1B (text generation) - stretch goal
- [ ] CLIP (multi-modal: image + text) - stretch goal
- [ ] Whisper-tiny (speech-to-text) - future
- [ ] Stable Diffusion (text-to-image) - future

**Model Management**:
- [x] Manual ONNX model loading from models/ directory
- [x] Model size detection and display in web UI
- [ ] Auto-download from HuggingFace Hub
- [ ] Model quantization pipeline (FP32 → INT8)
- [ ] Model versioning and metadata

### 🎯 Phase 4: VMM Service Daemon (Week 5-6)
**Goal**: Production-grade persistent service

- [ ] **Service Architecture**
  - [ ] gRPC or REST API server (C++)
  - [ ] Model registry and lifecycle management
  - [ ] Request queue and scheduling
  - [ ] Hot model reloading
  - [ ] Multi-client support
  - [ ] Connection pooling

### � Phase 5: Production Features (Week 7-8)
- [ ] Batch inference optimization
- [ ] Multi-GPU load balancing
- [ ] Request prioritization and QoS
- [ ] Prometheus metrics export
- [ ] OpenTelemetry tracing
- [ ] Docker containerization
- [ ] Kubernetes deployment manifests

### 🚀 Phase 6: Advanced Features (Future)
- [ ] NVIDIA backend (CUDA/TensorRT)
- [ ] AMD backend (ROCm)
- [ ] Model auto-optimization
- [ ] Multi-device inference pipelines
- [ ] Power/thermal management
- [ ] Dynamic model routing based on load

## 🎨 Current Demo: Multi-Modal AI Workbench

**Live Now**: http://localhost:8000

```
╔══════════════════════════════════════════════════════════╗
║  AI-VMM Dashboard                                       ║
╠══════════════════════════════════════════════════════════╣
║  🖥️  CPU: Intel Xeon w7-3455     [████████░░] 80%      ║
║  🎮  GPU: Arc B580 #1            [available]           ║
║  🎮  GPU: Arc B580 #2            [available]           ║
╠══════════════════════════════════════════════════════════╣
║  ✅ Working Models:                                     ║
║  📸 MobileNetV2 (14MB) - Image Classification          ║
║      CPU: ~194ms  |  GPU: ~198ms (similar on small)   ║
║  🎯 YOLOv8n (13MB) - Object Detection                  ║
║      CPU: 45ms    |  GPU: 28ms (1.6x speedup) ⚡      ║
╠══════════════════════════════════════════════════════════╣
║  🧪 Features:                                           ║
║  • Drag-drop image upload                              ║
║  • Example images (bus.jpg, zidane.jpg)                ║
║  • Real-time bounding box visualization               ║
║  • Device selection (Auto/CPU/GPU)                     ║
║  • Performance benchmarking                            ║
║  • VMM runtime diagnostics                             ║
╚══════════════════════════════════════════════════════════╝
```

**Next Demo Goals**:
- [ ] Live webcam → YOLOv8 real-time detection
- [ ] Larger models (YOLOv8m/l) showing >3x GPU speedup
- [ ] Multi-model pipeline (detection → classification)

---

## 🛠️ Build Requirements

### System Dependencies
- CMake 3.16+
- C++17 compatible compiler (GCC 8+, Clang 8+, MSVC 2019+)
- Python 3.8+ (for Python bindings and web API)

### Intel Backend Dependencies
- Intel oneAPI Toolkit 2025.0+ (DPC++, MKL, TBB)
- OpenVINO Toolkit 2024.0+
- Intel Graphics Compiler (for GPU support)

### NVIDIA Backend Dependencies (Future)
- CUDA Toolkit 11.0+
- cuDNN 8.0+ (recommended)
- TensorRT 8.0+ (optional)

### AMD Backend Dependencies (Future)
- ROCm 5.0+
- MIOpen (optional)

### Web Interface Dependencies (New)
- FastAPI 0.104+
- Uvicorn (ASGI server)
- Node.js 18+ and npm (for React dashboard)
- React 18+, TypeScript, Tailwind CSS

## Build Instructions

### Quick Start
```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Full Build with All Features
```bash
mkdir build && cd build
cmake .. \
    -DENABLE_INTEL_BACKEND=ON \
    -DENABLE_NVIDIA_BACKEND=ON \
    -DBUILD_PYTHON_BINDINGS=ON \
    -DBUILD_TESTS=ON
make -j$(nproc)
```

### Python Package Build
```bash
pip install -e . --verbose
```

## Development Environment Setup

### Intel oneAPI Setup
```bash
source /opt/intel/oneapi/setvars.sh
```

### CUDA Environment
```bash
export CUDA_ROOT=/usr/local/cuda
export PATH=$CUDA_ROOT/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_ROOT/lib64:$LD_LIBRARY_PATH
```

## Testing
```bash
cd build
ctest --verbose
```

## Python Development
```bash
# Install in development mode
pip install -e .[dev]

# Run Python tests
python -m pytest tests/python/
```

## Performance Profiling
- Intel VTune (for Intel backends)
- NVIDIA Nsight (for CUDA backends)  
- Perf/Valgrind (for general profiling)

## Code Style
- C++: Follow Google C++ Style Guide
- Python: Follow PEP 8
- Use clang-format for C++ formatting