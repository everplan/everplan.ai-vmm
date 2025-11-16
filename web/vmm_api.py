#!/usr/bin/env python3
"""
AI-VMM Web API Server
FastAPI-based REST API for interacting with the AI-VMM C++ library
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import subprocess
import json
import os
import tempfile
import base64
from pathlib import Path
import time
import threading
import atexit

app = FastAPI(
    title="AI-VMM API",
    description="REST API for AI Virtual Machine Manager",
    version="0.1.0"
)

# Enable CORS for web dashboard
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
STATIC_DIR = Path(__file__).parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Configuration
VMM_BUILD_DIR = Path("/root/everplan.ai-vmm/build")
BASIC_USAGE_BIN = VMM_BUILD_DIR / "examples/basic_usage/ai_vmm_basic_example"
PERF_BENCHMARK_BIN = VMM_BUILD_DIR / "examples/performance_comparison/ai_vmm_performance_comparison"
MODELS_DIR = Path("/root/everplan.ai-vmm/models")

# Global persistent inference server process
_yolov8_server = None
_yolov8_server_lock = None

# Pydantic models
class HardwareDevice(BaseModel):
    name: str
    type: str
    status: str = "available"

class ModelInfo(BaseModel):
    name: str
    path: str
    type: str
    size_mb: float

class InferenceRequest(BaseModel):
    model: str = "mobilenetv2"
    device: str = "auto"
    image_base64: Optional[str] = None

class InferenceResult(BaseModel):
    success: bool
    latency_ms: float
    device_used: str
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

class BenchmarkRequest(BaseModel):
    device: str = "all"
    iterations: int = 10

class VMMInfo(BaseModel):
    backends: Dict[str, Any]
    execution_providers: Dict[str, Any]
    hardware_capabilities: Dict[str, Any]
    system_info: Dict[str, Any]
    build_info: Dict[str, Any]

# Cache for hardware info
_hardware_cache = None
_cache_time = 0
CACHE_TTL = 10  # seconds


def get_or_start_yolov8_server():
    """Get or start the persistent YOLOv8 inference server"""
    global _yolov8_server, _yolov8_server_lock
    
    if _yolov8_server_lock is None:
        _yolov8_server_lock = threading.Lock()
    
    with _yolov8_server_lock:
        if _yolov8_server is None or _yolov8_server.poll() is not None:
            # Start new server process
            model_path = str(MODELS_DIR / "yolov8n.onnx")
            script_path = VMM_BUILD_DIR.parent / "src/backends/yolov8_server.py"
            
            venv_python = Path(__file__).parent / "venv" / "bin" / "python3"
            python_cmd = str(venv_python) if venv_python.exists() else "python3"
            
            _yolov8_server = subprocess.Popen(
                [python_cmd, str(script_path), model_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                cwd=str(VMM_BUILD_DIR.parent)
            )
            
            # Wait for server to be ready (read startup messages)
            while True:
                line = _yolov8_server.stdout.readline()
                if not line:
                    break
                msg = json.loads(line)
                if msg.get('status') == 'ready':
                    break
        
        return _yolov8_server


def stop_yolov8_server():
    """Stop the persistent YOLOv8 server"""
    global _yolov8_server
    if _yolov8_server and _yolov8_server.poll() is None:
        try:
            _yolov8_server.stdin.write(json.dumps({'command': 'exit'}) + '\n')
            _yolov8_server.stdin.flush()
            _yolov8_server.wait(timeout=5)
        except:
            _yolov8_server.kill()
        _yolov8_server = None


# Register cleanup on exit
atexit.register(stop_yolov8_server)


def get_hardware_info(force_refresh: bool = False) -> List[HardwareDevice]:
    """Get list of available hardware devices"""
    global _hardware_cache, _cache_time
    
    current_time = time.time()
    if not force_refresh and _hardware_cache and (current_time - _cache_time) < CACHE_TTL:
        return _hardware_cache
    
    try:
        # Run the performance benchmark with minimal iterations to get hardware info
        result = subprocess.run(
            [str(PERF_BENCHMARK_BIN), "--device", "all", "--iterations", "1"],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        devices = []
        device_counts = {}  # Track count of each device type for numbering
        
        # Combine stdout and stderr
        all_output = result.stderr + "\n" + result.stdout
        
        # Parse output for hardware information from "Available Hardware" section
        in_hardware_section = False
        for line in all_output.split('\n'):
            # Check if we're in the Available Hardware section
            if 'Available Hardware:' in line or '📋 Available Hardware:' in line:
                in_hardware_section = True
                continue
            
            # Exit hardware section when we hit the next section
            if in_hardware_section and ('═' in line or 'Running benchmarks' in line or '⏱️' in line):
                in_hardware_section = False
                continue
            
            # Parse hardware lines in the section
            if in_hardware_section and '•' in line:
                # Format: "  • Intel(R) Xeon(R) w7-3455 [CPU]"
                parts = line.split('•')[1].strip()
                if '[' in parts and ']' in parts:
                    name = parts.split('[')[0].strip()
                    dev_type = parts.split('[')[1].split(']')[0].strip()
                    
                    if name:
                        # Track device count for numbering duplicates
                        device_key = f"{name}_{dev_type}"
                        device_counts[device_key] = device_counts.get(device_key, 0) + 1
                        
                        # Add number suffix for duplicate devices (e.g., GPU #1, GPU #2)
                        display_name = name
                        if device_counts[device_key] > 1:
                            display_name = f"{name} #{device_counts[device_key]}"
                        
                        devices.append(HardwareDevice(
                            name=display_name,
                            type=dev_type,
                            status="available"
                        ))
        
        # Fallback: parse Intel Backend output if no devices found
        if not devices:
            device_counts = {}
            for line in all_output.split('\n'):
                # Parse CPU devices
                if 'Created CPU device:' in line:
                    name_match = line.split('device:')[-1].strip()
                    if name_match:
                        device_key = f"{name_match}_CPU"
                        device_counts[device_key] = device_counts.get(device_key, 0) + 1
                        
                        display_name = name_match
                        if device_counts[device_key] > 1:
                            display_name = f"{name_match} #{device_counts[device_key]}"
                        
                        devices.append(HardwareDevice(
                            name=display_name,
                            type="CPU",
                            status="available"
                        ))
                        
                # Parse GPU devices  
                elif 'Found Intel GPU via PCIe:' in line:
                    # Extract GPU name - format: "Found Intel GPU via PCIe: Intel Arc A580M (Battlemage) (ID: e20b)"
                    gpu_info = line.split(':')[-1].strip()
                    # Remove ID part if present
                    if '(ID:' in gpu_info:
                        gpu_name = gpu_info.split('(ID:')[0].strip()
                    else:
                        gpu_name = gpu_info
                    
                    if gpu_name:
                        device_key = f"{gpu_name}_GPU"
                        device_counts[device_key] = device_counts.get(device_key, 0) + 1
                        
                        display_name = gpu_name
                        if device_counts[device_key] > 1:
                            display_name = f"{gpu_name} #{device_counts[device_key]}"
                        
                        devices.append(HardwareDevice(
                            name=display_name,
                            type="Intel Arc GPU",
                            status="available"
                        ))
        
        # Final fallback if still no devices
        if not devices:
            devices = [
                HardwareDevice(name="CPU (Auto-detected)", type="CPU", status="available"),
                HardwareDevice(name="GPU (Auto-detected)", type="GPU", status="available")
            ]
        
        _hardware_cache = devices
        _cache_time = current_time
        
        return devices
        
    except subprocess.TimeoutExpired:
        return [HardwareDevice(name="CPU (timeout)", type="CPU", status="available")]
    except Exception as e:
        # Return minimal hardware info on error
        return [
            HardwareDevice(name=f"CPU (error: {str(e)[:20]})", type="CPU", status="available"),
            HardwareDevice(name="GPU", type="GPU", status="unknown")
        ]


def get_available_models() -> List[ModelInfo]:
    """Get list of available models"""
    models = []
    
    if MODELS_DIR.exists():
        for model_file in MODELS_DIR.glob("*.onnx"):
            size_mb = model_file.stat().st_size / (1024 * 1024)
            models.append(ModelInfo(
                name=model_file.stem,
                path=str(model_file),
                type="ONNX",
                size_mb=round(size_mb, 2)
            ))
    
    # Add default model if directory is empty
    if not models:
        models.append(ModelInfo(
            name="mobilenetv2",
            path=str(MODELS_DIR / "mobilenetv2.onnx"),
            type="ONNX",
            size_mb=13.96
        ))
    
    return models


@app.get("/")
async def root():
    """Serve the dashboard"""
    dashboard_path = STATIC_DIR / "index.html"
    if dashboard_path.exists():
        return FileResponse(dashboard_path)
    return {
        "service": "AI-VMM API",
        "version": "0.1.0",
        "status": "running",
        "dashboard": "Not found - static files missing",
        "endpoints": {
            "hardware": "/api/hardware",
            "models": "/api/models",
            "infer": "/api/infer",
            "benchmark": "/api/benchmark"
        }
    }


@app.get("/api/hardware", response_model=List[HardwareDevice])
async def list_hardware(refresh: bool = False):
    """
    Get list of available hardware devices
    
    - **refresh**: Force refresh of hardware detection
    """
    return get_hardware_info(force_refresh=refresh)


@app.get("/api/models", response_model=List[ModelInfo])
async def list_models():
    """Get list of available AI models"""
    return get_available_models()


@app.get("/api/model-info/{model_name}")
async def get_model_metadata(model_name: str):
    """Get detailed metadata for a specific model"""
    model_info_path = Path(__file__).parent.parent / "models" / "model_info.json"
    
    if model_info_path.exists():
        with open(model_info_path, 'r') as f:
            all_models = json.load(f)
            if model_name in all_models:
                return all_models[model_name]
    
    raise HTTPException(status_code=404, detail=f"Model metadata not found for: {model_name}")


@app.post("/api/infer")
async def run_inference(
    image: UploadFile = File(...), 
    model: str = Form("mobilenetv2"), 
    device: str = Form("auto")
):
    """
    Run inference on uploaded image
    
    - **image**: Image file to process
    - **model**: Model to use (mobilenetv2, yolov8n)
    - **device**: Target device (cpu, gpu, auto)
    """
    start_time = time.time()
    
    try:
        # Read image and convert to base64
        content = await image.read()
        image_base64 = base64.b64encode(content).decode('utf-8')
        
        # Select inference script based on model
        if model == "yolov8n":
            # Use persistent server for faster inference
            server = get_or_start_yolov8_server()
            
            # Map device to provider
            device_arg = "GPU" if device.lower() in ["gpu", "auto"] else "CPU"
            
            # Send inference request to server
            request = json.dumps({
                'command': 'infer',
                'image': image_base64,
                'device': device_arg
            }) + '\n'
            
            inference_start = time.time()
            server.stdin.write(request)
            server.stdin.flush()
            
            # Read response
            response_line = server.stdout.readline()
            latency_ms = (time.time() - inference_start) * 1000
            
            # Parse output
            try:
                output = json.loads(response_line)
                if output.get('success'):
                    detections = output.get('detections', [])
                    providers = output.get('providers', [])
                    device_used = "GPU (OpenVINO)" if "OpenVINOExecutionProvider" in providers else "CPU"
                    
                    return InferenceResult(
                        success=True,
                        latency_ms=round(latency_ms, 2),
                        device_used=device_used,
                        results={
                            "type": "detection",
                            "detections": detections,
                            "count": len(detections),
                            "model": "YOLOv8n"
                        }
                    )
                else:
                    raise HTTPException(status_code=500, detail=output.get('error', 'Unknown error'))
            except json.JSONDecodeError:
                raise HTTPException(status_code=500, detail=f"Failed to parse output: {result.stdout}")
        
        elif model in ["mobilenetv2", "resnet50"]:
            # Classification models - use resnet50_inference.py
            # Save uploaded image temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                tmp_file.write(content)
                tmp_path = tmp_file.name
            
            try:
                # Get paths
                models_dir = Path(__file__).parent.parent / "models"
                inference_script = Path(__file__).parent.parent / "src" / "backends" / "resnet50_inference.py"
                model_path = models_dir / f"{model}.onnx"
                python_exe = Path(__file__).parent / "venv" / "bin" / "python3"
                
                # Map device
                device_arg = "GPU" if device.lower() in ["gpu", "auto"] else "CPU"
                
                # Run inference
                inference_start = time.time()
                result = subprocess.run(
                    [str(python_exe), str(inference_script), str(model_path), tmp_path, device_arg],
                    capture_output=True,
                    text=True,
                    timeout=10
                )
                latency_ms = (time.time() - inference_start) * 1000
                
                if result.returncode != 0:
                    raise HTTPException(status_code=500, detail=f"Inference failed: {result.stderr}")
                
                # Parse JSON output
                output = json.loads(result.stdout)
                
                # Format device name
                providers = output.get('providers', [])
                device_used = "GPU (OpenVINO)" if "OpenVINOExecutionProvider" in providers else "CPU"
                
                return InferenceResult(
                    success=True,
                    latency_ms=round(latency_ms, 2),
                    device_used=device_used,
                    results={
                        "type": "classification",
                        "predictions": output.get('predictions', [])[:5],  # Top 5
                        "model": model
                    }
                )
                
            finally:
                # Cleanup temporary file
                os.unlink(tmp_path)
        
        else:
            # Unknown model
            raise HTTPException(status_code=400, detail=f"Unsupported model: {model}")
            
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Inference timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")


@app.post("/api/benchmark")
async def run_benchmark(request: BenchmarkRequest):
    """
    Run performance benchmark
    
    - **device**: Device to benchmark (cpu, gpu, all)
    - **iterations**: Number of iterations (default: 10)
    """
    try:
        # Run from the performance_comparison directory where models/ symlink exists
        benchmark_dir = PERF_BENCHMARK_BIN.parent
        
        result = subprocess.run(
            [
                str(PERF_BENCHMARK_BIN),
                "--device", request.device,
                "--iterations", str(request.iterations)
            ],
            capture_output=True,
            text=True,
            timeout=60,
            cwd=str(benchmark_dir)  # Run from the correct directory
        )
        
        # Parse benchmark results
        results = []
        in_results_section = False
        
        # Combine stdout and stderr (benchmark output goes to both)
        all_output = result.stdout + "\n" + result.stderr
        
        # Parse using regex to handle variable whitespace and line wrapping
        import re
        
        # Look for lines with Intel device names followed by numbers
        # Pattern: Intel...device_name (2+ spaces) num num num num inf/s
        pattern = r'(Intel.*?)\s{2,}(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+inf/s'
        
        for match in re.finditer(pattern, all_output):
            device_name = match.group(1).strip()
            avg_ms = float(match.group(2))
            min_ms = float(match.group(3))
            max_ms = float(match.group(4))
            throughput_val = float(match.group(5))
            
            results.append({
                "device": device_name,
                "avg_ms": avg_ms,
                "min_ms": min_ms,
                "max_ms": max_ms,
                "throughput": f"{throughput_val} inf/s"
            })
        
        return {
            "success": True,
            "device_filter": request.device,
            "iterations": request.iterations,
            "results": results
        }
        
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=504, detail="Benchmark timeout")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Benchmark failed: {str(e)}")


@app.get("/api/stats")
async def get_stats():
    """Get system statistics and metrics"""
    try:
        # Get basic system info
        import platform
        import psutil
        
        return {
            "system": {
                "os": platform.system(),
                "platform": platform.platform(),
                "cpu_count": psutil.cpu_count(),
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "memory_total_gb": round(psutil.virtual_memory().total / (1024**3), 2),
                "memory_used_gb": round(psutil.virtual_memory().used / (1024**3), 2),
                "memory_percent": psutil.virtual_memory().percent
            },
            "vmm": {
                "version": "0.1.0",
                "build_dir": str(VMM_BUILD_DIR),
                "models_loaded": len(get_available_models()),
                "hardware_devices": len(get_hardware_info())
            }
        }
    except Exception as e:
        return {
            "error": str(e),
            "vmm": {
                "version": "0.1.0",
                "status": "degraded"
            }
        }


def get_vmm_info() -> Dict[str, Any]:
    """Gather comprehensive VMM runtime information"""
    import platform
    import re
    
    info = {
        "backends": {},
        "execution_providers": {},
        "hardware_capabilities": {},
        "system_info": {},
        "build_info": {}
    }
    
    # 1. Detect Backend Support & Versions
    try:
        result = subprocess.run(
            [str(PERF_BENCHMARK_BIN), "--device", "all", "--iterations", "1"],
            capture_output=True, text=True, timeout=30,
            cwd=str(PERF_BENCHMARK_BIN.parent)
        )
        output = result.stdout + "\n" + result.stderr
        
        # Parse backend versions from output
        backends = {}
        
        # OpenVINO version
        openvino_match = re.search(r'OpenVINO.*?(\d+\.\d+\.\d+)', output, re.IGNORECASE)
        if openvino_match:
            backends['openvino'] = {
                'status': 'available',
                'version': openvino_match.group(1)
            }
        
        # Intel backend
        if 'Intel Backend' in output or 'intel backend' in output.lower():
            backends['intel'] = {
                'status': 'available',
                'devices_found': output.count('Created CPU device:') + output.count('Found Intel GPU')
            }
        
        # ONNX Runtime version
        if 'ONNX' in output:
            backends['onnx_runtime'] = {'status': 'available'}
        
        info['backends'] = backends
        
    except Exception as e:
        info['backends'] = {'error': str(e)}
    
    # 2. ONNX Runtime Execution Providers
    try:
        import onnxruntime as ort
        available_providers = ort.get_available_providers()
        
        providers = {}
        for provider in available_providers:
            providers[provider] = {
                'status': 'available',
                'active': True
            }
        
        # Add common providers that might be missing
        all_known = ['CPUExecutionProvider', 'CUDAExecutionProvider', 
                     'TensorrtExecutionProvider', 'OpenVINOExecutionProvider',
                     'DmlExecutionProvider', 'ROCMExecutionProvider']
        
        for provider in all_known:
            if provider not in providers:
                providers[provider] = {
                    'status': 'not_available',
                    'active': False
                }
        
        info['execution_providers'] = {
            'runtime_version': ort.__version__,
            'providers': providers
        }
        
    except Exception as e:
        info['execution_providers'] = {'error': str(e)}
    
    # 3. Hardware Capabilities
    try:
        hw_caps = {}
        
        # CPU info
        try:
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
                
            # Extract CPU model
            model_match = re.search(r'model name\s+:\s+(.+)', cpuinfo)
            if model_match:
                hw_caps['cpu_model'] = model_match.group(1).strip()
            
            # CPU flags
            flags_match = re.search(r'flags\s+:\s+(.+)', cpuinfo)
            if flags_match:
                flags = flags_match.group(1).split()
                hw_caps['cpu_features'] = {
                    'avx': 'avx' in flags,
                    'avx2': 'avx2' in flags,
                    'avx512f': 'avx512f' in flags,
                    'avx512_vnni': 'avx512_vnni' in flags,
                    'amx_tile': 'amx_tile' in flags,
                    'amx_int8': 'amx_int8' in flags,
                    'amx_bf16': 'amx_bf16' in flags,
                }
        except Exception as e:
            hw_caps['cpu_error'] = str(e)
        
        # GPU info from lspci
        try:
            result = subprocess.run(['lspci'], capture_output=True, text=True)
            gpus = []
            for line in result.stdout.split('\n'):
                if 'VGA' in line or 'Display' in line or '3D controller' in line:
                    gpus.append(line.split(': ', 1)[-1] if ': ' in line else line)
            hw_caps['gpus'] = gpus
        except:
            pass
        
        # Memory info
        try:
            import psutil
            mem = psutil.virtual_memory()
            hw_caps['memory'] = {
                'total_gb': round(mem.total / (1024**3), 2),
                'available_gb': round(mem.available / (1024**3), 2)
            }
        except:
            pass
        
        info['hardware_capabilities'] = hw_caps
        
    except Exception as e:
        info['hardware_capabilities'] = {'error': str(e)}
    
    # 4. System Information
    try:
        info['system_info'] = {
            'os': platform.system(),
            'os_version': platform.version(),
            'platform': platform.platform(),
            'python_version': platform.python_version(),
            'architecture': platform.machine(),
            'kernel': platform.release()
        }
        
        # Driver versions
        drivers = {}
        
        # Intel GPU driver
        try:
            result = subprocess.run(['modinfo', 'i915'], capture_output=True, text=True)
            version_match = re.search(r'version:\s+(.+)', result.stdout)
            if version_match:
                drivers['intel_graphics'] = version_match.group(1).strip()
        except:
            pass
        
        # NVIDIA driver
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                                  capture_output=True, text=True, timeout=2)
            if result.returncode == 0:
                drivers['nvidia'] = result.stdout.strip()
        except:
            pass
        
        info['system_info']['drivers'] = drivers
        
    except Exception as e:
        info['system_info'] = {'error': str(e)}
    
    # 5. Build Information
    try:
        build_info = {
            'version': '0.1.0',
            'binaries': {
                'basic_usage': str(BASIC_USAGE_BIN.exists()),
                'performance_comparison': str(PERF_BENCHMARK_BIN.exists())
            }
        }
        
        # Try to get build timestamp
        if PERF_BENCHMARK_BIN.exists():
            import datetime
            mtime = PERF_BENCHMARK_BIN.stat().st_mtime
            build_info['binary_timestamp'] = datetime.datetime.fromtimestamp(mtime).isoformat()
        
        info['build_info'] = build_info
        
    except Exception as e:
        info['build_info'] = {'error': str(e)}
    
    return info


@app.get("/api/vmm-info")
async def vmm_info():
    """
    Get comprehensive VMM runtime information
    
    Returns detailed information about:
    - Backend support (OpenVINO, ONNX Runtime, etc.)
    - Execution providers status
    - Hardware capabilities (CPU features, GPUs, memory)
    - System environment (OS, drivers, Python)
    - Build information
    """
    try:
        return get_vmm_info()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to gather VMM info: {str(e)}")


@app.get("/examples/images/{filename}")
async def get_example_image(filename: str):
    """
    Serve example images for testing
    
    Args:
        filename: Name of the image file (e.g., "bus.jpg", "zidane.jpg")
    
    Returns:
        FileResponse with the image
    """
    examples_dir = Path(__file__).parent.parent / "examples" / "images"
    file_path = examples_dir / filename
    
    # Security: prevent directory traversal
    if not file_path.resolve().is_relative_to(examples_dir.resolve()):
        raise HTTPException(status_code=403, detail="Access denied")
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"Example image '{filename}' not found")
    
    return FileResponse(file_path)


if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting AI-VMM Web API Server...")
    print(f"📁 Build directory: {VMM_BUILD_DIR}")
    print(f"📦 Models directory: {MODELS_DIR}")
    print(f"🔧 Binaries:")
    print(f"   - Basic usage: {BASIC_USAGE_BIN}")
    print(f"   - Benchmark: {PERF_BENCHMARK_BIN}")
    print()
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
