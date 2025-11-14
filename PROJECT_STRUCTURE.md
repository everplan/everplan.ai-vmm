# AI VMM Project Structure

```
ai-vmm/
├── CMakeLists.txt              # Main build configuration
├── README.md                   # Project overview and usage
├── DEVELOPMENT.md              # Development setup and guidelines
├── ai-vmm.md                   # Original concept document
│
├── include/ai_vmm/             # Public C++ headers
│   ├── ai_vmm.hpp             # Main include (includes all others)
│   ├── types.hpp              # Core types and enums
│   ├── compute_backend.hpp    # Hardware abstraction interface
│   └── vmm.hpp                # Main VMM class interface
│
├── src/                       # Source code implementation
│   ├── core/                  # Core VMM functionality
│   │   ├── CMakeLists.txt
│   │   ├── vmm.cpp           # Main VMM implementation
│   │   ├── tensor.cpp        # Tensor operations
│   │   ├── model.cpp         # Model representation
│   │   └── hardware_discovery.cpp
│   │
│   ├── hal/                   # Hardware Abstraction Layer
│   │   ├── CMakeLists.txt
│   │   └── backend_registry.cpp
│   │
│   ├── backends/              # Hardware-specific implementations
│   │   ├── intel/            # Intel CPU, iGPU, dGPU, NPU
│   │   │   ├── CMakeLists.txt
│   │   │   ├── cpu_backend.cpp
│   │   │   ├── igpu_backend.cpp
│   │   │   ├── arc_backend.cpp
│   │   │   └── npu_backend.cpp
│   │   │
│   │   ├── nvidia/           # NVIDIA CUDA GPUs
│   │   │   ├── CMakeLists.txt
│   │   │   ├── cuda_backend.cpp
│   │   │   └── tensorrt_backend.cpp
│   │   │
│   │   └── amd/             # AMD ROCm GPUs (future)
│   │       ├── CMakeLists.txt
│   │       └── rocm_backend.cpp
│   │
│   ├── optimization/          # Model optimization pipeline
│   │   ├── CMakeLists.txt
│   │   ├── model_optimizer.cpp
│   │   ├── quantization.cpp
│   │   └── kernel_fusion.cpp
│   │
│   ├── scheduling/            # Workload scheduling
│   │   ├── CMakeLists.txt
│   │   ├── workload_scheduler.cpp
│   │   └── resource_manager.cpp
│   │
│   ├── memory/               # Cross-vendor memory management
│   │   ├── CMakeLists.txt
│   │   ├── memory_manager.cpp
│   │   └── unified_memory.cpp
│   │
│   └── python/               # Python bindings (pybind11)
│       ├── CMakeLists.txt
│       ├── python_bindings.cpp
│       └── tensor_numpy.cpp
│
├── python/                   # Python package
│   ├── ai_vmm/
│   │   ├── __init__.py
│   │   ├── utils.py
│   │   └── examples.py
│   ├── setup.py
│   └── requirements.txt
│
├── examples/                 # Example applications
│   ├── CMakeLists.txt
│   ├── basic_usage/
│   │   ├── main.cpp
│   │   └── CMakeLists.txt
│   ├── multi_model_serving/
│   └── performance_comparison/
│
├── tests/                    # Test suite
│   ├── CMakeLists.txt
│   ├── test_main.cpp
│   ├── test_tensor.cpp
│   ├── test_vmm_basic.cpp
│   ├── test_hardware_discovery.cpp
│   ├── backend_tests/
│   └── python/
│
├── docs/                     # Documentation
│   ├── api_reference/
│   ├── tutorials/
│   └── architecture.md
│
├── scripts/                  # Build and utility scripts  
│   ├── build.sh
│   ├── install_deps.sh
│   └── run_benchmarks.sh
│
└── third_party/             # External dependencies (if needed)
    ├── onnx/
    └── pybind11/
```

## Key Design Principles

### Modular Architecture
- **Core**: Framework-agnostic foundation
- **HAL**: Hardware abstraction without vendor dependencies  
- **Backends**: Vendor-specific implementations
- **Optimization**: Model transformation pipeline
- **Scheduling**: Resource management and workload distribution

### Multi-Vendor Support
- Intel: oneAPI, OpenVINO, Level Zero
- NVIDIA: CUDA, cuDNN, TensorRT
- AMD: ROCm (future)
- Unified memory management across vendors

### Extensibility
- Plugin-based backend system
- Model format agnostic (ONNX, OpenVINO IR, etc.)
- Optimization passes can be chained
- Custom hardware support via backend interface

### Developer Experience
- Simple C++ and Python APIs
- Automatic hardware discovery
- Intelligent model optimization
- Comprehensive error handling and debugging

This structure provides a solid foundation for building a production-ready AI VMM that can scale from edge devices to data center deployments while maintaining optimal performance across diverse hardware platforms.