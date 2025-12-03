#!/usr/bin/env python3
"""
Download and convert AI models for the VMM
"""

import os
import sys
import argparse
from pathlib import Path

def download_tinyllama():
    """Download and convert TinyLlama to OpenVINO IR format"""
    print("🔥 Downloading TinyLlama-1.1B-Chat-v1.0...")
    print("This will download ~2GB and convert to OpenVINO IR format...")
    
    try:
        from optimum.intel.openvino import OVModelForCausalLM
        from transformers import AutoTokenizer
        
        model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        output_dir = "models/tinyllama_openvino"
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        print(f"📥 Downloading {model_id}...")
        
        # Download and convert to OpenVINO IR
        model = OVModelForCausalLM.from_pretrained(
            model_id,
            export=True,  # Convert to OpenVINO IR
            compile=False  # Don't compile yet, we'll do that per-device
        )
        
        # Save the model
        print(f"💾 Saving model to {output_dir}...")
        model.save_pretrained(output_dir)
        
        # Download tokenizer
        print("📝 Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.save_pretrained(output_dir)
        
        print(f"\n✅ TinyLlama model successfully downloaded!")
        print(f"📁 Location: {output_dir}")
        
        # Show file sizes
        print(f"\n📊 Files in {output_dir}:")
        total_size = 0
        for f in Path(output_dir).iterdir():
            if f.is_file() and not f.name.endswith('.py'):
                size_mb = f.stat().st_size / (1024*1024)
                total_size += size_mb
                print(f"  {f.name:40s} {size_mb:8.1f} MB")
        
        print(f"\n📦 Total size: {total_size:.1f} MB")
        return True
        
    except ImportError as e:
        print(f"❌ Missing dependencies: {e}")
        print("Please install: pip install optimum[openvino]")
        return False
    except Exception as e:
        print(f"❌ Error downloading model: {e}")
        return False

def download_mobilenet():
    """Download MobileNetV2 ONNX model"""
    print("🔥 Downloading MobileNetV2 ONNX model...")
    
    try:
        import urllib.request
        
        url = "https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx"
        output_path = "models/mobilenetv2.onnx"
        
        print(f"📥 Downloading from {url}...")
        urllib.request.urlretrieve(url, output_path)
        
        size_mb = Path(output_path).stat().st_size / (1024*1024)
        print(f"✅ MobileNetV2 downloaded: {size_mb:.1f} MB")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading MobileNetV2: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Download AI models for VMM")
    parser.add_argument("--model", choices=["all", "tinyllama", "mobilenet"], 
                       default="all", help="Which model(s) to download")
    parser.add_argument("--force", action="store_true", 
                       help="Force re-download even if model exists")
    
    args = parser.parse_args()
    
    print("🚀 AI-VMM Model Downloader")
    print("=" * 40)
    
    # Change to script directory
    os.chdir(Path(__file__).parent)
    
    # Create models directory
    Path("models").mkdir(exist_ok=True)
    
    success = True
    
    if args.model in ["all", "tinyllama"]:
        if not Path("models/tinyllama_openvino/openvino_model.xml").exists() or args.force:
            success &= download_tinyllama()
        else:
            print("✅ TinyLlama already exists (use --force to re-download)")
    
    if args.model in ["all", "mobilenet"]:
        if not Path("models/mobilenetv2.onnx").exists() or args.force:
            success &= download_mobilenet()
        else:
            print("✅ MobileNetV2 already exists (use --force to re-download)")
    
    if success:
        print(f"\n🎉 All models downloaded successfully!")
        print(f"\n📁 Available models:")
        for model_file in Path("models").rglob("*.onnx"):
            size_mb = model_file.stat().st_size / (1024*1024)
            print(f"  {model_file}: {size_mb:.1f} MB")
        for xml_file in Path("models").rglob("openvino_model.xml"):
            print(f"  {xml_file.parent} (OpenVINO IR)")
    else:
        print(f"\n❌ Some downloads failed")
        sys.exit(1)

if __name__ == "__main__":
    main()