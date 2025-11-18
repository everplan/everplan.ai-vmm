#!/usr/bin/env python3
"""
TinyLlama Text Generation Inference
Uses ONNX Runtime with OpenVINO backend for Intel GPU acceleration
"""

import onnxruntime as ort
import numpy as np
import json
import time
import sys
from pathlib import Path
from transformers import AutoTokenizer

def get_llm_providers(device='CPU'):
    """
    Get optimal execution providers for LLM inference based on hardware
    
    Args:
        device: 'CPU', 'GPU', 'intel_cpu', 'intel_gpu', 'nvidia_gpu', 'auto'
        
    Returns:
        List of ONNX Runtime execution providers in priority order
    """
    device = device.upper()
    
    # Normalize device names
    if device in ['GPU', 'INTEL_GPU', 'ARC']:
        return [
            ('OpenVINOExecutionProvider', {
                'device_type': 'GPU',
                'enable_dynamic_shapes': True,
                'precision': 'FP16'  # FP16 faster on GPU
            }),
            'CPUExecutionProvider'  # Fallback
        ]
    
    elif device in ['NVIDIA_GPU', 'CUDA']:
        return [
            ('TensorrtExecutionProvider', {
                'trt_fp16_enable': True,
                'trt_engine_cache_enable': True
            }),
            ('CUDAExecutionProvider', {
                'device_id': 0
            }),
            'CPUExecutionProvider'
        ]
    
    elif device in ['AMD_GPU', 'ROCM']:
        return [
            ('ROCMExecutionProvider', {
                'device_id': 0
            }),
            'CPUExecutionProvider'
        ]
    
    elif device == 'AUTO':
        # Intelligent auto-selection based on available providers
        available = ort.get_available_providers()
        
        if 'OpenVINOExecutionProvider' in available:
            # Prefer OpenVINO GPU if available, else CPU
            return ['OpenVINOExecutionProvider', 'CPUExecutionProvider']
        elif 'TensorrtExecutionProvider' in available:
            return ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
        elif 'CUDAExecutionProvider' in available:
            return ['CUDAExecutionProvider', 'CPUExecutionProvider']
        else:
            return ['CPUExecutionProvider']
    
    else:  # CPU or default
        return [
            ('OpenVINOExecutionProvider', {
                'device_type': 'CPU',
                'enable_dynamic_shapes': True
            }),
            'CPUExecutionProvider'
        ]

class LLMInference:
    def __init__(self, model_path, tokenizer_path, device='CPU'):
        """Initialize LLM inference session with multi-backend support"""
        self.device = device
        
        # Load tokenizer
        print(f"Loading tokenizer from {tokenizer_path}...", file=sys.stderr)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # Get optimal providers for this device
        providers = get_llm_providers(device)
        print(f"Requested device: {device}", file=sys.stderr)
        print(f"Provider configuration: {providers}", file=sys.stderr)
        
        # Set up ONNX Runtime session
        print(f"Loading model from {model_path}...", file=sys.stderr)
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        
        actual_providers = self.session.get_providers()
        print(f"Model loaded. Active providers: {actual_providers}", file=sys.stderr)
        
        # Log which backend is actually being used
        if 'OpenVINOExecutionProvider' in actual_providers:
            print(f"✓ Using OpenVINO backend (Intel optimized)", file=sys.stderr)
        elif 'TensorrtExecutionProvider' in actual_providers:
            print(f"✓ Using TensorRT backend (NVIDIA optimized)", file=sys.stderr)
        elif 'CUDAExecutionProvider' in actual_providers:
            print(f"✓ Using CUDA backend (NVIDIA)", file=sys.stderr)
        elif 'ROCMExecutionProvider' in actual_providers:
            print(f"✓ Using ROCm backend (AMD optimized)", file=sys.stderr)
        else:
            print(f"✓ Using CPU backend", file=sys.stderr)
        
        # Get input/output names
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        
        print(f"Model inputs: {self.input_names[:3]}...", file=sys.stderr)  # Show first 3
        
    def generate(self, prompt, max_length=100, temperature=0.7, top_p=0.9, stream=False):
        """
        Generate text from prompt
        
        Args:
            prompt: Input text prompt
            max_length: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling parameter
            stream: Whether to yield tokens as generated
            
        Returns:
            Generated text dict (or generator if stream=True)
        """
        if stream:
            return self._generate_stream(prompt, max_length, temperature, top_p)
        else:
            return self._generate_complete(prompt, max_length, temperature, top_p)
    
    def _generate_complete(self, prompt, max_length, temperature, top_p):
        """Generate complete text (non-streaming)"""
        # Tokenize input
        input_ids = self.tokenizer.encode(prompt, return_tensors='np')
        
        if isinstance(input_ids, list):
            input_ids = np.array([input_ids])
        
        start_time = time.time()
        generated_tokens = []
        first_token_time = None
        
        for i in range(max_length):
            # Prepare inputs
            inputs = {
                'input_ids': input_ids.astype(np.int64)
            }
            
            # Run inference
            outputs = self.session.run(self.output_names, inputs)
            
            # Get logits for next token
            logits = outputs[0][0, -1, :]  # Last token logits
            
            # Apply temperature
            logits = logits / temperature
            
            # Apply softmax
            probs = np.exp(logits) / np.sum(np.exp(logits))
            
            # Top-p (nucleus) sampling
            sorted_indices = np.argsort(probs)[::-1]
            sorted_probs = probs[sorted_indices]
            cumulative_probs = np.cumsum(sorted_probs)
            
            # Find cutoff index for top_p
            cutoff_index = np.searchsorted(cumulative_probs, top_p)
            top_indices = sorted_indices[:cutoff_index+1]
            top_probs = probs[top_indices]
            top_probs = top_probs / np.sum(top_probs)  # Renormalize
            
            # Sample from top-p distribution
            next_token = np.random.choice(top_indices, p=top_probs)
            
            if first_token_time is None:
                first_token_time = time.time()
            
            # Check for end of sequence
            if next_token == self.tokenizer.eos_token_id:
                break
            
            generated_tokens.append(next_token)
            
            # Append token to input for next iteration
            input_ids = np.concatenate([input_ids, [[next_token]]], axis=1)
        
        # Calculate metrics
        total_time = time.time() - start_time
        tokens_per_sec = len(generated_tokens) / total_time if total_time > 0 else 0
        time_to_first_token = (first_token_time - start_time) * 1000 if first_token_time else 0
        
        # Decode full text
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return {
            'text': generated_text,
            'tokens_generated': len(generated_tokens),
            'total_time_sec': total_time,
            'tokens_per_sec': tokens_per_sec,
            'time_to_first_token_ms': time_to_first_token,
            'providers': self.session.get_providers(),
            'device': self.device
        }
    
    def _generate_stream(self, prompt, max_length, temperature, top_p):
        """Generate text with streaming (yields tokens as they're generated)"""
        # Tokenize input
        input_ids = self.tokenizer.encode(prompt, return_tensors='np')
        
        generated_tokens = []
        start_time = time.time()
        first_token_time = None
        
        for i in range(max_length):
            # Prepare inputs
            inputs = {
                'input_ids': input_ids.astype(np.int64)
            }
            
            # Run inference
            outputs = self.session.run(self.output_names, inputs)
            
            # Get logits for next token
            logits = outputs[0][0, -1, :]  # Last token logits
            
            # Apply temperature
            logits = logits / temperature
            
            # Apply softmax
            probs = np.exp(logits) / np.sum(np.exp(logits))
            
            # Top-p (nucleus) sampling
            sorted_indices = np.argsort(probs)[::-1]
            sorted_probs = probs[sorted_indices]
            cumulative_probs = np.cumsum(sorted_probs)
            
            # Find cutoff index for top_p
            cutoff_index = np.searchsorted(cumulative_probs, top_p)
            top_indices = sorted_indices[:cutoff_index+1]
            top_probs = probs[top_indices]
            top_probs = top_probs / np.sum(top_probs)  # Renormalize
            
            # Sample from top-p distribution
            next_token = np.random.choice(top_indices, p=top_probs)
            
            if first_token_time is None:
                first_token_time = time.time()
            
            # Check for end of sequence
            if next_token == self.tokenizer.eos_token_id:
                break
            
            generated_tokens.append(next_token)
            
            # Decode and yield token
            token_text = self.tokenizer.decode([next_token])
            yield token_text
            
            # Append token to input for next iteration
            input_ids = np.concatenate([input_ids, [[next_token]]], axis=1)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Run LLM inference with ONNX Runtime')
    parser.add_argument('--model', required=True, help='Path to ONNX model file')
    parser.add_argument('--prompt', required=True, help='Text prompt for generation')
    parser.add_argument('--max-tokens', type=int, default=50, help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--top-p', type=float, default=0.9, help='Top-p (nucleus) sampling')
    parser.add_argument('--device', default='CPU', choices=['CPU', 'GPU', 'auto'], 
                       help='Device to run inference on')
    
    args = parser.parse_args()
    
    # Determine tokenizer path (assume it's in the same directory as model)
    model_path = Path(args.model)
    tokenizer_path = model_path.parent
    
    if not model_path.exists():
        print(json.dumps({'error': f'Model not found: {model_path}'}))
        sys.exit(1)
    
    # Initialize model
    try:
        llm = LLMInference(str(model_path), str(tokenizer_path), device=args.device)
    except Exception as e:
        print(json.dumps({'error': f'Failed to load model: {str(e)}'}))
        sys.exit(1)
    
    # Generate text
    try:
        result = llm.generate(
            args.prompt, 
            max_length=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )
        
        # Output JSON result
        print(json.dumps(result, indent=2))
    except Exception as e:
        print(json.dumps({'error': f'Generation failed: {str(e)}'}))
        sys.exit(1)

if __name__ == "__main__":
    main()
