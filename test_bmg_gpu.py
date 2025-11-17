#!/usr/bin/env python3
"""
Intel Arc GPU (Battlemage) Test Script using IPEX XPU
Run this script periodically to check when GPU support becomes available
"""

import sys

def test_ipex_xpu():
    """Test if IPEX XPU can detect and use Intel Arc GPUs"""
    
    print("=" * 70)
    print("Intel Arc B580 (Battlemage) GPU Detection Test")
    print("=" * 70)
    
    try:
        import torch
        import intel_extension_for_pytorch as ipex
    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("\nTo install IPEX XPU:")
        print("  cd /root/everplan.ai-vmm")
        print("  source web/venv/bin/activate")
        print("  pip install torch==2.5.1+cxx11.abi torchvision==0.20.1+cxx11.abi \\")
        print("    intel-extension-for-pytorch==2.5.10+xpu \\")
        print("    --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/")
        return False
    
    print(f"\n✅ PyTorch version: {torch.__version__}")
    print(f"✅ IPEX version: {ipex.__version__}")
    
    # Check XPU availability
    try:
        xpu_available = torch.xpu.is_available()
        print(f"\nXPU available: {xpu_available}")
        
        if xpu_available:
            device_count = torch.xpu.device_count()
            print(f"XPU device count: {device_count}")
            
            if device_count > 0:
                print("\n🎉 SUCCESS! Intel Arc GPUs are now accessible!")
                print("\nDetected GPUs:")
                for i in range(device_count):
                    name = torch.xpu.get_device_name(i)
                    print(f"  [{i}] {name}")
                
                # Test GPU compute
                print("\n" + "-" * 70)
                print("Running GPU Compute Test...")
                
                import time
                start = time.time()
                x = torch.randn(2000, 2000).to('xpu:0')
                y = torch.randn(2000, 2000).to('xpu:0')
                z = torch.matmul(x, y)
                torch.xpu.synchronize()
                elapsed = time.time() - start
                
                print(f"✓ Matrix multiply (2000x2000): {elapsed*1000:.2f}ms")
                print(f"✓ Result shape: {z.shape}")
                print(f"✓ Device: {z.device}")
                
                # Test model inference
                print("\n" + "-" * 70)
                print("Testing Simple Model Inference...")
                
                model = torch.nn.Sequential(
                    torch.nn.Linear(1000, 500),
                    torch.nn.ReLU(),
                    torch.nn.Linear(500, 10)
                ).to('xpu:0')
                
                input_data = torch.randn(32, 1000).to('xpu:0')
                
                with torch.no_grad():
                    start = time.time()
                    output = model(input_data)
                    torch.xpu.synchronize()
                    elapsed = time.time() - start
                
                print(f"✓ Model forward pass: {elapsed*1000:.2f}ms")
                print(f"✓ Output shape: {output.shape}")
                
                print("\n" + "=" * 70)
                print("🚀 Intel Arc B580 GPUs are FULLY FUNCTIONAL!")
                print("=" * 70)
                print("\nNext steps:")
                print("  1. Update web UI to enable GPU backend")
                print("  2. Benchmark LLM performance on GPU vs CPU")
                print("  3. Test dual-GPU parallel inference")
                print("  4. Update GPU_STATUS.md with results")
                
                return True
            else:
                print("\n⚠️  XPU available but no devices detected")
                return False
        else:
            print("\n❌ XPU NOT available")
            print("\nThis means the Intel compute runtime (Level Zero) is not yet")
            print("compatible with the Xe kernel driver used by Battlemage GPUs.")
            print("\nExpected timeline: Q1-Q2 2026")
            print("\nTo check for updates:")
            print("  - apt update && apt upgrade")
            print("  - Check https://github.com/intel/compute-runtime")
            print("  - Re-run this test script")
            return False
            
    except Exception as e:
        print(f"\n❌ Error during XPU detection: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_ipex_xpu()
    sys.exit(0 if success else 1)
