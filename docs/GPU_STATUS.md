# Intel Arc GPU Support Status

## Hardware

- 2× Intel Arc B580 Battlemage GPUs
- Device IDs: 0000:18:00.0, 0000:ae:00.0
- Kernel Driver: **Xe** (new Intel GPU driver for Battlemage/Alchemist)

## Current Status

**GPU Compute NOT Working** - Battlemage Not Yet Supported in Compute Runtimes

### Investigation Results (November 16, 2025)

**IPEX XPU Installation Attempted:**
- ✅ PyTorch 2.5.1+cxx11.abi (XPU build) installed
- ✅ Intel Extension for PyTorch 2.5.10+xpu installed
- ❌ `torch.xpu.is_available()` returns `False`
- ❌ `torch.xpu.device_count()` returns `0`
- ❌ `sycl-ls` tool crashes (core dumped)

### Root Cause

The Intel Arc B580 (Battlemage) GPUs use the new **Xe kernel driver**, which is NOT YET SUPPORTED by Intel's compute runtime stack (Level Zero, OpenCL). Current status:

- ✅ Kernel driver (xe): Working - GPUs detected at `/dev/dri/renderD128`, `/dev/dri/renderD129`
- ✅ Display/Graphics: Working (firmware loaded: `i915/bmg_dmc.bin v2.6`)
- ❌ Level Zero runtime: Does not enumerate XPU devices (count = 0)
- ❌ OpenCL compute: Only detects CPU, no GPU devices  
- ❌ SYCL/DPC++: Crashes when querying devices
- ❌ OpenVINO GPU plugin: Cannot detect GPUs
- ❌ IPEX XPU: Cannot detect GPUs (depends on Level Zero)

### What's Installed

```bash
# Intel GPU Runtime Packages
intel-level-zero-gpu    1.3.30049.10-950~24.04
libze1                  1.21.9.0-1136~24.04
libigc1                 1.0.17791.16-1032~24.04
libigdfcl1              1.0.17791.16-1032~24.04
libigdgmm12             22.7.2-1135~24.04

# Current compute runtime was built for i915, not Xe
```

### Detection Results

**OpenCL (clinfo):**
```
Platform: Intel(R) OpenCL
Number of GPU devices: 0
Number of CPU devices: 1 (Xeon w7-3455)
```

**OpenVINO:**
```python
from openvino import Core
core = Core()
core.available_devices  # Returns: ['CPU']
# GPU.0 and GPU.1 not detected
```

**Level Zero:**
```
# Level Zero library loads but finds no GPU devices
# Issue: intel-level-zero-gpu built for i915, needs Xe support
```

## Workaround - CPU Performance

While GPU support is pending, CPU performance is excellent:

- **Intel Xeon w7-3455**: 24 cores (48 threads) with AMX/VNNI
- **TinyLlama-1.1B**: **45 tokens/sec** average on CPU
- **YOLOv8n**: 45ms inference on CPU (vs 28ms when GPU was working with ONNX RT)

## Path Forward

### Option 1: Wait for Intel Compute Runtime Update (Recommended)

Intel is actively developing Xe driver support. Monitor:
- https://github.com/intel/compute-runtime
- Ubuntu Intel GPU PPA updates

Expected timeline: Q1-Q2 2026 for stable Xe support in compute stack.

### Option 2: Use ONNX Runtime OpenVINO EP (Partial Workaround)

The YOLOv8 model **was working** on GPU earlier using ONNX Runtime with OpenVINOExecutionProvider. This suggests:

1. ONNX Runtime's OpenVINO EP might have better Xe support
2. The issue is specific to native OpenVINO (used by Optimum Intel for LLMs)

**Investigation needed:**
- Check if ONNX Runtime OpenVINO EP can still access GPU
- If yes, consider adapting LLM inference to use ONNX Runtime instead of native OpenVINO

### Option 3: Fallback to i915 Driver (Not Recommended)

Could try forcing i915 instead of Xe, but:
- ❌ May lose Battlemage-specific optimizations
- ❌ i915 may not fully support BMG hardware
- ❌ Requires kernel parameter changes and reboot
- ❌ Goes against Intel's direction (Xe is the future)

## Testing GPU Support

To check if GPU support has been enabled:

```bash
# Check OpenVINO
python3 << 'EOF'
from openvino import Core
core = Core()
print("Devices:", core.available_devices)
# Should show: ['CPU', 'GPU.0', 'GPU.1']
EOF

# Check OpenCL
clinfo | grep "Device Type.*GPU"
# Should show GPU devices

# Check Level Zero
ls -la /dev/dri/renderD*
# Should have renderD128, renderD129 (already present)
```

## Performance Target

When GPU support is working:

- **Expected LLM speedup**: 2-3x over CPU (based on other OpenVINO GPU benchmarks)
- **Target**: 90-135 tokens/sec on Arc B580 GPU
- **Memory**: 12GB VRAM per GPU, sufficient for TinyLlama-1.1B

## Current Recommendations

1. **For Demo/Development**: Use CPU (45 tokens/sec is acceptable for demo)
2. **For Production**: Monitor Intel compute runtime updates
3. **For Benchmarking**: Document CPU performance, note GPU as "coming soon"
4. **For Value Prop**: Focus on multi-backend architecture, not specific GPU performance

## Related Issues

- Ubuntu 24.10 (Oracular) ships with Xe driver by default
- Intel compute runtime in Ubuntu repos may lag behind kernel
- Newer Intel GPU platforms (Arc, Battlemage) require latest runtime versions

## Last Tested

- Date: November 16, 2025
- OpenVINO: 2025.3.0
- Kernel: 6.11.0-29-generic
- GPU Driver: Xe (in-tree)
- Compute Runtime: intel-level-zero-gpu 1.3.30049
