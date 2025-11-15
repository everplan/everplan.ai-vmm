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

### 🚧 Phase 2: Service Architecture (IN PROGRESS - Week 1-2)
**Goal**: Transform VMM from library into a persistent daemon/service

- [x] **Web UI + REST API** ✅ COMPLETED
  - [x] FastAPI backend wrapping VMM C++ library
  - [x] Modern web dashboard with real-time metrics
  - [x] Live inference playground (image upload → classification)
  - [x] Hardware monitoring and visualization
  - [x] Performance benchmarking interface
  - [x] System statistics (CPU, memory)
  - [ ] WebSocket support for streaming results (next)

- [ ] **VMM Service Daemon** (Next Sprint)
  - [ ] gRPC or REST API server (C++)
  - [ ] Model registry and lifecycle management
  - [ ] Request queue and scheduling
  - [ ] Hot model reloading
  - [ ] Multi-client support

**Web Dashboard Running**: http://localhost:8000  
**API Documentation**: http://localhost:8000/docs

### 🎯 Phase 3: Model Zoo Expansion (Week 3-4)
**Goal**: Prove versatility across model types

**Priority Models** (ONNX format):
- [x] MobileNetV2 (image classification)
- [ ] YOLOv8 (object detection) - **HIGH PRIORITY** for sexy demo
- [ ] TinyLlama-1.1B (text generation)
- [ ] CLIP (multi-modal: image + text)
- [ ] Whisper-tiny (speech-to-text)
- [ ] Stable Diffusion (text-to-image) - stretch goal

**Model Management**:
- [ ] Auto-download from HuggingFace Hub
- [ ] Model quantization pipeline (FP32 → INT8)
- [ ] Model versioning and metadata

### 🎬 Phase 4: Production Features (Week 5-6)
- [ ] Batch inference optimization
- [ ] Multi-GPU load balancing
- [ ] Request prioritization and QoS
- [ ] Prometheus metrics export
- [ ] OpenTelemetry tracing
- [ ] Docker containerization

### 🌟 Phase 5: Advanced Features (Future)
- [ ] NVIDIA backend (CUDA/TensorRT)
- [ ] AMD backend (ROCm)
- [ ] Model auto-optimization
- [ ] Multi-device inference pipelines
- [ ] Power/thermal management

## 🎨 Target Demo: Multi-Modal AI Workbench

**The Showcase**: Live web interface demonstrating heterogeneous hardware orchestration

```
╔══════════════════════════════════════════════════════════╗
║  AI-VMM Dashboard                                       ║
╠══════════════════════════════════════════════════════════╣
║  🖥️  CPU: Intel Xeon w7-3455     [████████░░] 80%      ║
║  🎮  GPU: Arc B580 #1            [██████░░░░] 60%      ║
║  🎮  GPU: Arc B580 #2            [███░░░░░░░] 30%      ║
╠══════════════════════════════════════════════════════════╣
║  Live Inference:                                        ║
║  [Webcam] → YOLOv8 (GPU) → Real-time Detection         ║
║  [Text]   → TinyLlama (CPU) → Text Generation          ║
║  [Image]  → CLIP (iGPU) → Multi-modal Search           ║
╚══════════════════════════════════════════════════════════╝
```

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