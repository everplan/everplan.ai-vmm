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

class LLMInference:
    def __init__(self, model_path, tokenizer_path, device='CPU'):
        """Initialize LLM inference session"""
        self.device = device
        
        # Load tokenizer
        print(f"Loading tokenizer from {tokenizer_path}...", file=sys.stderr)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        # Set up ONNX Runtime session
        print(f"Loading model from {model_path}...", file=sys.stderr)
        if device.upper() == 'GPU':
            providers = ['OpenVINOExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        print(f"Model loaded. Providers: {self.session.get_providers()}", file=sys.stderr)
        
        # Get input/output names
        self.input_names = [inp.name for inp in self.session.get_inputs()]
        self.output_names = [out.name for out in self.session.get_outputs()]
        
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
            Generated text or generator if stream=True
        """
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
            
            # Decode and yield if streaming
            if stream:
                token_text = self.tokenizer.decode([next_token])
                yield token_text
            
            # Append token to input for next iteration
            input_ids = np.concatenate([input_ids, [[next_token]]], axis=1)
        
        # Calculate metrics
        total_time = time.time() - start_time
        tokens_per_sec = len(generated_tokens) / total_time if total_time > 0 else 0
        time_to_first_token = (first_token_time - start_time) * 1000 if first_token_time else 0
        
        # Decode full text
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        if not stream:
            return {
                'text': generated_text,
                'tokens_generated': len(generated_tokens),
                'total_time_sec': total_time,
                'tokens_per_sec': tokens_per_sec,
                'time_to_first_token_ms': time_to_first_token,
                'providers': self.session.get_providers(),
                'device': self.device
            }

def main():
    if len(sys.argv) < 3:
        print("Usage: python llm_inference.py <model_path> <tokenizer_path> [prompt] [device] [max_length]")
        sys.exit(1)
    
    model_path = sys.argv[1]
    tokenizer_path = sys.argv[2]
    prompt = sys.argv[3] if len(sys.argv) > 3 else "The capital of France is"
    device = sys.argv[4] if len(sys.argv) > 4 else 'CPU'
    max_length = int(sys.argv[5]) if len(sys.argv) > 5 else 50
    
    # Initialize model
    llm = LLMInference(model_path, tokenizer_path, device=device)
    
    # Generate text
    result = llm.generate(prompt, max_length=max_length)
    
    # Output JSON result
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
