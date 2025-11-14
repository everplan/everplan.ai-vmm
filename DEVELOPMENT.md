# Development Configuration

## Build Requirements

### System Dependencies
- CMake 3.16+
- C++17 compatible compiler (GCC 8+, Clang 8+, MSVC 2019+)
- Python 3.8+ (for Python bindings)

### Intel Backend Dependencies
- Intel oneAPI Toolkit (optional but recommended)
- OpenVINO Toolkit 2023.0+
- Intel Graphics Compiler (for GPU support)

### NVIDIA Backend Dependencies  
- CUDA Toolkit 11.0+
- cuDNN 8.0+ (recommended)
- TensorRT 8.0+ (optional)

### AMD Backend Dependencies (Future)
- ROCm 5.0+
- MIOpen (optional)

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