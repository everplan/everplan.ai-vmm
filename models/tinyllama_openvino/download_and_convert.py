#!/usr/bin/env python3
"""
Download and convert TinyLlama to OpenVINO IR format
"""
from optimum.intel.openvino import OVModelForCausalLM
from transformers import AutoTokenizer
import shutil

model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
output_dir = "."

print(f"Downloading {model_id}...")
print("This will download ~2GB and convert to OpenVINO IR format...")

# Download and convert in one step
# Note: load_in_8bit requires NNCF for quantization
# For now, use FP16 (still good performance, easier)
model = OVModelForCausalLM.from_pretrained(
    model_id,
    export=True,  # Convert to OpenVINO IR
    compile=False  # Don't compile yet, we'll do that per-device
)

# Save the model
model.save_pretrained(output_dir)
print(f"✓ Model saved to {output_dir}")

# Download tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.save_pretrained(output_dir)
print(f"✓ Tokenizer saved to {output_dir}")

print("\nModel conversion complete!")
print(f"Files in {output_dir}:")
import os
for f in os.listdir(output_dir):
    if not f.endswith('.py'):
        size = os.path.getsize(f) / (1024*1024)
        print(f"  {f:40s} {size:8.1f} MB")
