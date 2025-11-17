#!/usr/bin/env python3
"""
LLM Inference using Optimum Intel + OpenVINO
Handles all KV-cache complexity automatically via OVModelForCausalLM
"""

import sys
import json
import time
from pathlib import Path
from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoTokenizer
import torch


class LLMInference:
    """
    LLM inference using Optimum Intel (OpenVINO backend)
    
    Supports:
    - Automatic KV-cache management
    - Multi-device: CPU (AMX), GPU (XMX)
    - Temperature/top_p sampling
    - Performance metrics
    """
    
    def __init__(self, model_path, device='CPU'):
        """
        Initialize LLM model
        
        Args:
            model_path: Path to OpenVINO IR model directory
            device: 'CPU' or 'GPU'
        """
        self.model_path = Path(model_path)
        self.device = device
        
        print(f"Loading LLM from {self.model_path}...", file=sys.stderr)
        print(f"Target device: {device}", file=sys.stderr)
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
        
        # Load model with OpenVINO backend
        # This handles all KV-cache complexity!
        self.model = OVModelForCausalLM.from_pretrained(
            str(self.model_path),
            device=device,
            ov_config={"PERFORMANCE_HINT": "LATENCY"}  # Optimize for low latency
        )
        
        print(f"✓ LLM loaded on {device}", file=sys.stderr)
        print(f"✓ Model uses KV-cache optimization", file=sys.stderr)
        
    def generate(self, prompt, max_tokens=50, temperature=0.7, top_p=0.9):
        """
        Generate text from prompt
        
        Args:
            prompt: Input text prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.1-2.0)
            top_p: Nucleus sampling parameter (0.1-1.0)
            
        Returns:
            dict with generated text and metadata
        """
        start_time = time.time()
        
        print(f"\nGenerating with config:", file=sys.stderr)
        print(f"  max_tokens: {max_tokens}", file=sys.stderr)
        print(f"  temperature: {temperature}", file=sys.stderr)
        print(f"  top_p: {top_p}", file=sys.stderr)
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_length = inputs.input_ids.shape[1]
        
        # Generate with KV-cache (handled automatically by Optimum)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True  # Enable KV-cache
            )
        
        # Decode output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove input prompt from output
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):]
        
        # Calculate metrics
        total_time = time.time() - start_time
        output_length = outputs.shape[1]
        tokens_generated = output_length - input_length
        tokens_per_sec = tokens_generated / total_time if total_time > 0 else 0
        
        print(f"\n✓ Generation complete:", file=sys.stderr)
        print(f"  {tokens_generated} tokens in {total_time:.2f}s", file=sys.stderr)
        print(f"  {tokens_per_sec:.1f} tokens/sec", file=sys.stderr)
        
        return {
            'text': generated_text.strip(),
            'tokens_generated': tokens_generated,
            'total_time_sec': round(total_time, 3),
            'tokens_per_sec': round(tokens_per_sec, 2),
            'device': self.device,
            'backend': 'Optimum Intel (OpenVINO)',
            'model_path': str(self.model_path)
        }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='LLM inference with OpenVINO GenAI')
    parser.add_argument('--model', required=True, help='Path to OpenVINO IR model directory')
    parser.add_argument('--prompt', required=True, help='Text prompt for generation')
    parser.add_argument('--max-tokens', type=int, default=50, help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
    parser.add_argument('--top-p', type=float, default=0.9, help='Top-p (nucleus) sampling')
    parser.add_argument('--device', default='CPU', choices=['CPU', 'GPU', 'AUTO'], 
                       help='Device to run inference on')
    
    args = parser.parse_args()
    
    try:
        # Initialize model
        llm = LLMInference(args.model, device=args.device)
        
        # Generate text
        result = llm.generate(
            args.prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p
        )
        
        # Output JSON result
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        error_result = {'error': f'Generation failed: {str(e)}'}
        print(json.dumps(error_result))
        sys.exit(1)


if __name__ == "__main__":
    main()
