#!/usr/bin/env python3
"""Test PyTorch XPU inference on Intel Arc B580 GPUs"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time

def main():
    print("=" * 60)
    print("PyTorch XPU Intel Arc B580 Inference Test")
    print("=" * 60)
    
    # Check GPU availability
    print(f"\n✅ PyTorch Version: {torch.__version__}")
    print(f"✅ XPU Available: {torch.xpu.is_available()}")
    print(f"✅ XPU Device Count: {torch.xpu.device_count()}")
    
    if torch.xpu.device_count() > 0:
        for i in range(torch.xpu.device_count()):
            print(f"   - Device {i}: {torch.xpu.get_device_name(i)}")
    
    # Load model
    print("\n📥 Loading TinyLlama model from HuggingFace...")
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    start = time.time()
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    print(f"✅ Tokenizer loaded in {time.time() - start:.2f}s")
    
    start = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    model = model.to('xpu:0')  # Use first B580 GPU
    print(f"✅ Model loaded in {time.time() - start:.2f}s")
    
    # Test inference
    print("\n🤖 Running inference...")
    prompt = "Q: What is the capital of France?\nA:"
    
    start = time.time()
    inputs = tokenizer(prompt, return_tensors="pt").to('xpu:0')
    
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
    
    inference_time = time.time() - start
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    tokens_generated = len(outputs[0]) - len(inputs.input_ids[0])
    tokens_per_sec = tokens_generated / inference_time
    
    print(f"\n📝 Prompt: {prompt}")
    print(f"📝 Response: {response}")
    print(f"\n⚡ Performance:")
    print(f"   - Inference Time: {inference_time:.3f}s")
    print(f"   - Tokens Generated: {tokens_generated}")
    print(f"   - Throughput: {tokens_per_sec:.2f} tokens/s")
    print(f"   - GPU: {torch.xpu.get_device_name(0)}")
    
    print("\n✅ Test completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    main()
