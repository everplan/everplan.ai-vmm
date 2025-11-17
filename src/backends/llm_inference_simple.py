#!/usr/bin/env python3
"""
Simple LLM inference stub for testing UI
TODO: Implement full KV-cache handling for decoder models
"""

import json
import sys
import argparse
import time

def main():
    parser = argparse.ArgumentParser(description='Simple LLM inference (stub)')
    parser.add_argument('--model', required=True, help='Path to ONNX model file')
    parser.add_argument('--prompt', required=True, help='Text prompt for generation')
    parser.add_argument('--max-tokens', type=int, default=50, help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=1.0, help='Sampling temperature')
    parser.add_argument('--top-p', type=float, default=0.9, help='Top-p sampling')
    parser.add_argument('--device', default='CPU', help='Device')
    
    args = parser.parse_args()
    
    start_time = time.time()
    
    # Simulate text generation (placeholder for full implementation)
    # This demonstrates the UI/API flow while KV-cache implementation is in progress
    
    completions = {
        "The capital of France is": " Paris, the City of Light. It's known for the Eiffel Tower, Louvre Museum, and rich cultural heritage.",
        "Once upon a time": " there was a small village nestled in the mountains. The villagers lived in harmony with nature.",
        "Explain quantum computing": " in simple terms: Quantum computers use quantum bits (qubits) that can be in multiple states simultaneously, enabling parallel processing of complex problems.",
        "Write a haiku": " about autumn:\nLeaves fall gently down\nGolden hues paint the landscape\nNature's art unfolds"
    }
    
    # Find matching completion or generate generic response
    generated_text = None
    for key, value in completions.items():
        if args.prompt.startswith(key):
            generated_text = value
            break
    
    if generated_text is None:
        # Generic completion
        generated_text = f" [Response to: '{args.prompt[:50]}...'] This is a placeholder response while we implement full KV-cache handling for the TinyLlama decoder architecture."
    
    # Simulate some processing time
    time.sleep(0.5)
    
    total_time = time.time() - start_time
    tokens_generated = len(generated_text.split())
    tokens_per_sec = tokens_generated / total_time if total_time > 0 else 0
    
    result = {
        'text': generated_text,
        'tokens_generated': tokens_generated,
        'total_time_sec': round(total_time, 3),
        'tokens_per_sec': round(tokens_per_sec, 2),
        'time_to_first_token_ms': round(0.1 * 1000, 2),
        'providers': ['CPUExecutionProvider (stub)'],
        'device': args.device,
        'note': 'This is a stub implementation. Full KV-cache decoder support in progress.'
    }
    
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    main()
