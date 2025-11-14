#!/usr/bin/env python3
"""
ONNX Runtime inference wrapper for AI VMM
Provides a Python interface for model inference that can be called from C++
"""
import sys
import json
import numpy as np
import os

# Suppress ONNX Runtime warnings BEFORE importing onnxruntime
os.environ['ORT_LOGGING_LEVEL'] = '3'  # Error level only - suppresses device discovery warnings

import onnxruntime as ort
from pathlib import Path

def load_model(model_path, device_type="CPU"):
    """Load ONNX model with specified device"""
    try:
        # Debug: Print available providers
        print(f"[ONNX Debug] Available providers: {ort.get_available_providers()}")
        print(f"[ONNX Debug] Requested device type: {device_type}")
        print(f"[ONNX Debug] Note: DRM device discovery warnings are harmless - OpenVINO provides direct hardware access")
        
        # Set up provider based on device type
        providers = []
        if device_type == "CPU":
            providers = ['CPUExecutionProvider']
        elif device_type == "GPU":
            # Try to use actual GPU provider if available
            available = ort.get_available_providers()
            if 'OpenVINOExecutionProvider' in available:
                # OpenVINO can accelerate on Intel GPUs
                providers = ['OpenVINOExecutionProvider', 'CPUExecutionProvider']
                print(f"[ONNX Debug] Using OpenVINO provider for Intel GPU acceleration")
            elif 'CUDAExecutionProvider' in available:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                print(f"[ONNX Debug] Using CUDA provider")
            else:
                providers = ['CPUExecutionProvider']
                print(f"[ONNX Debug] No GPU providers available, falling back to CPU")
        elif device_type == "NPU":
            # For NPU, try OpenVINO first (supports Intel NPU)
            available = ort.get_available_providers()
            if 'OpenVINOExecutionProvider' in available:
                providers = ['OpenVINOExecutionProvider', 'CPUExecutionProvider']
                print(f"[ONNX Debug] Using OpenVINO provider for Intel NPU acceleration")
            else:
                providers = ['CPUExecutionProvider']
                print(f"[ONNX Debug] No NPU providers available, falling back to CPU")
        else:
            providers = ['CPUExecutionProvider']
        
        print(f"[ONNX Debug] Using providers: {providers}")
        session = ort.InferenceSession(model_path, providers=providers)
        
        # Get model info
        input_info = {
            'name': session.get_inputs()[0].name,
            'shape': session.get_inputs()[0].shape,
            'type': session.get_inputs()[0].type
        }
        
        output_info = {
            'name': session.get_outputs()[0].name,
            'shape': session.get_outputs()[0].shape,
            'type': session.get_outputs()[0].type
        }
        
        return {
            'success': True,
            'session_id': id(session),
            'input': input_info,
            'output': output_info,
            'providers': session.get_providers()
        }, session
        
    except Exception as e:
        return {'success': False, 'error': str(e)}, None

def run_inference(session, input_data):
    """Run inference with given input data"""
    try:
        input_name = session.get_inputs()[0].name
        input_tensor = np.array(input_data, dtype=np.float32)
        
        # Run inference
        results = session.run(None, {input_name: input_tensor})
        
        return {
            'success': True,
            'output': results[0].tolist()
        }
        
    except Exception as e:
        return {'success': False, 'error': str(e)}

def main():
    """Command line interface for inference"""
    if len(sys.argv) < 3:
        print(json.dumps({'success': False, 'error': 'Usage: python3 onnx_inference.py <model_path> <command> [args]'}))
        return
    
    model_path = sys.argv[1]
    command = sys.argv[2]
    
    if command == "load":
        device_type = sys.argv[3] if len(sys.argv) > 3 else "CPU"
        result, session = load_model(model_path, device_type)
        print(json.dumps(result))
        
    elif command == "test":
        # Quick test with dummy data
        result, session = load_model(model_path, "CPU")
        if result['success']:
            # Create dummy input matching the expected shape
            input_shape = result['input']['shape']
            # Replace dynamic dimensions with 1
            actual_shape = [1 if isinstance(dim, str) else dim for dim in input_shape]
            dummy_input = np.random.randn(*actual_shape).astype(np.float32)
            
            inference_result = run_inference(session, dummy_input)
            result['test_inference'] = inference_result
        
        print(json.dumps(result))
        
    else:
        print(json.dumps({'success': False, 'error': f'Unknown command: {command}'}))

if __name__ == "__main__":
    main()