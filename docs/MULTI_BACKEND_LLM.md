# VMM Multi-Backend Architecture for LLMs

## Overview

VMM provides a **hardware-agnostic abstraction layer** that routes LLM inference to the optimal backend based on available hardware. Unlike Ollama (which uses llama.cpp for CPU/CUDA) or vendor-specific solutions, VMM supports **any hardware** through **standardized ONNX models**.

## Architecture Layers

```
┌─────────────────────────────────────────────────────────────┐
│                     VMM API Layer                           │
│  - Unified REST/gRPC interface                              │
│  - Model metadata and routing                               │
│  - Request queuing and batching                             │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────┴───────────────────────────────────────┐
│            Hardware Abstraction Layer (HAL)                 │
│  - Automatic hardware detection                             │
│  - Provider selection (OpenVINO, TensorRT, DirectML, etc.)  │
│  - Session management and caching                           │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┬─────────────┐
        ↓             ↓             ↓             ↓
┌──────────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│  Intel CPU   │ │Intel GPU │ │NVIDIA GPU│ │  AMD GPU │
│              │ │          │ │          │ │          │
│  OpenVINO EP │ │OpenVINO  │ │TensorRT  │ │  ROCm EP │
│  + AMX/VNNI  │ │   EP     │ │    EP    │ │          │
└──────────────┘ └──────────┘ └──────────┘ └──────────┘
```

## LLM-Specific Considerations

### 1. **Provider Selection for Different Hardware**

```python
def get_llm_providers(device='auto'):
    """
    Select optimal execution providers for LLM inference
    based on available hardware and model characteristics
    """
    if device == 'auto':
        device = detect_best_hardware()
    
    provider_configs = {
        'intel_cpu': [
            ('OpenVINOExecutionProvider', {
                'device_type': 'CPU',
                'enable_dynamic_shapes': True,
                'cache_dir': '/tmp/ov_cache'
            }),
            'CPUExecutionProvider'
        ],
        
        'intel_gpu': [
            ('OpenVINOExecutionProvider', {
                'device_type': 'GPU',
                'enable_dynamic_shapes': True,
                'precision': 'FP16',  # FP16 faster on GPU
                'cache_dir': '/tmp/ov_cache'
            }),
            'CPUExecutionProvider'  # Fallback
        ],
        
        'nvidia_gpu': [
            ('TensorrtExecutionProvider', {
                'trt_max_workspace_size': 4 * 1024 * 1024 * 1024,  # 4GB
                'trt_fp16_enable': True,
                'trt_engine_cache_enable': True
            }),
            ('CUDAExecutionProvider', {
                'device_id': 0,
                'arena_extend_strategy': 'kSameAsRequested'
            }),
            'CPUExecutionProvider'
        ],
        
        'amd_gpu': [
            ('ROCMExecutionProvider', {
                'device_id': 0
            }),
            'CPUExecutionProvider'
        ]
    }
    
    return provider_configs.get(device, ['CPUExecutionProvider'])
```

### 2. **Autoregressive Generation with KV-Cache**

LLMs require special handling for efficient generation:

```python
class LLMSession:
    def __init__(self, model_path, device='auto'):
        self.providers = get_llm_providers(device)
        
        # Load model with optimized providers
        self.session = ort.InferenceSession(
            model_path,
            providers=self.providers,
            sess_options=self._get_session_options()
        )
        
        # Initialize KV cache for autoregressive generation
        self.kv_cache = None
        
    def generate_with_cache(self, input_ids, max_length=100):
        """
        Efficient generation using KV-cache to avoid recomputing
        past tokens on each iteration
        """
        past_key_values = self._initialize_kv_cache()
        
        for i in range(max_length):
            # Only pass new token, reuse cached key/values
            if i == 0:
                # First iteration: full prompt
                inputs = self._prepare_inputs(input_ids, past_key_values)
            else:
                # Subsequent: only new token + cached KV
                inputs = self._prepare_inputs(
                    input_ids[:, -1:],  # Last token only
                    past_key_values
                )
            
            # Run inference
            outputs = self.session.run(None, inputs)
            
            # Extract logits and updated KV cache
            logits = outputs[0]
            past_key_values = self._extract_kv_cache(outputs)
            
            # Sample next token
            next_token = self._sample(logits, temperature=0.7)
            input_ids = np.concatenate([input_ids, [[next_token]]], axis=1)
            
            yield next_token
```

### 3. **Hardware-Specific Optimizations**

Different backends get different optimizations:

**Intel CPU (Xeon w7-3455)**
- AMX (Advanced Matrix Extensions) for INT8 quantized models
- AVX-512 VNNI for faster INT8 inference
- OpenVINO CPU plugin with weight sharing

**Intel GPU (Arc Battlemage)**
- OpenVINO GPU plugin with FP16 precision
- Unified memory for zero-copy between CPU/GPU
- XMX (Xe Matrix Extensions) for matrix operations

**NVIDIA GPU (Future)**
- TensorRT optimization with FP16/INT8
- CUDA kernel fusion
- Tensor Core acceleration

**AMD GPU (Future)**
- ROCm execution provider
- MIOpen for optimized kernels

## Real-World Example

### Current YOLOv8 Multi-Backend Pattern

```python
# From yolov8_server.py - this pattern extends to LLMs
class YOLOv8Server:
    def __init__(self, model_path):
        # Pre-load BOTH CPU and GPU sessions
        self.sessions = {}
        
        self.sessions['cpu'] = ort.InferenceSession(
            model_path, 
            providers=['CPUExecutionProvider']
        )
        
        self.sessions['gpu'] = ort.InferenceSession(
            model_path,
            providers=['OpenVINOExecutionProvider', 'CPUExecutionProvider']
        )
    
    def infer(self, image, device='GPU'):
        # Route to appropriate session
        session = self.sessions.get(device.lower(), self.sessions['cpu'])
        outputs = session.run(None, inputs)
        return outputs
```

### Future LLM Multi-Backend Pattern

```python
class LLMServer:
    def __init__(self, model_path):
        self.sessions = {}
        
        # Pre-load sessions for each available hardware
        self.sessions['cpu'] = ort.InferenceSession(
            model_path,
            providers=get_llm_providers('intel_cpu')
        )
        
        if has_intel_gpu():
            self.sessions['intel_gpu'] = ort.InferenceSession(
                model_path,
                providers=get_llm_providers('intel_gpu')
            )
        
        if has_nvidia_gpu():
            self.sessions['nvidia_gpu'] = ort.InferenceSession(
                model_path,
                providers=get_llm_providers('nvidia_gpu')
            )
    
    def generate(self, prompt, device='auto'):
        # Auto-select best device if not specified
        if device == 'auto':
            device = self._select_best_device()
        
        session = self.sessions.get(device, self.sessions['cpu'])
        
        # Tokenize
        input_ids = self.tokenizer.encode(prompt)
        
        # Generate with selected backend
        for token in self._autoregressive_generate(session, input_ids):
            yield token
```

## Comparison with Other Solutions

| Feature | VMM | Ollama | TensorRT | OpenVINO |
|---------|-----|--------|----------|----------|
| **Model Format** | ONNX (universal) | GGUF (llama.cpp) | TensorRT engines | IR (OpenVINO) |
| **Intel CPU** | ✓ OpenVINO + AMX | ✓ llama.cpp | ✗ | ✓ |
| **Intel GPU** | ✓ OpenVINO | ✗ (no Arc support) | ✗ | ✓ |
| **NVIDIA GPU** | ✓ TensorRT EP | ✓ CUDA (via llama.cpp) | ✓ | ✗ |
| **AMD GPU** | ✓ ROCm EP (future) | ✓ ROCm (via llama.cpp) | ✗ | ✗ |
| **Vision Models** | ✓ Same platform | ✗ | ✓ | ✓ |
| **Auto Backend** | ✓ Intelligent routing | ✗ Manual | ✗ | ✗ |

## VMM's Key Advantages

1. **Single Model Format (ONNX)**
   - No need to convert models for each backend
   - GGUF (Ollama) → TensorRT → OpenVINO IR → ONNX
   - One model runs on ALL hardware

2. **Unified API**
   - Same API for LLMs, vision, speech, multi-modal
   - Ollama requires different API than vision tools

3. **Intelligent Hardware Selection**
   - Automatically picks best backend for workload
   - Small models → CPU (lower latency)
   - Large models → GPU (higher throughput)
   - Batch requests → GPU (parallelism)

4. **Vendor Independence**
   - Not locked to NVIDIA (TensorRT) or Intel (OpenVINO)
   - Easy to add new backends (Apple Silicon, Qualcomm NPU)

## Implementation Roadmap for LLM Backend

### Phase 1: Single Backend (Current)
- ✅ ONNX model loaded
- ✅ CPU inference via CPUExecutionProvider
- 🔄 Basic generation (needs KV-cache handling)

### Phase 2: Multi-Backend Support
- [ ] OpenVINO CPU (with AMX acceleration)
- [ ] OpenVINO GPU (Arc Battlemage)
- [ ] Automatic provider selection
- [ ] Provider fallback chain

### Phase 3: Optimization
- [ ] KV-cache management for efficient generation
- [ ] Batched inference support
- [ ] Quantization-aware inference (INT8)
- [ ] Dynamic sequence length handling

### Phase 4: Advanced Features
- [ ] Multi-GPU inference (model parallelism)
- [ ] Pipeline parallelism (split across devices)
- [ ] Mixed-precision inference
- [ ] Model sharding for large models

## Conclusion

VMM's architecture allows **the same ONNX LLM model** to run efficiently on:
- Intel Xeon CPU (with AMX/VNNI acceleration)
- Intel Arc GPU (with OpenVINO GPU plugin)
- NVIDIA GPU (with TensorRT)
- AMD GPU (with ROCm)

All through a **single API**, with **automatic hardware selection**, across **all model types** (not just LLMs).

This is VMM's core differentiator vs specialized tools!
