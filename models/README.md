# AI-VMM Model Zoo

This directory contains diverse AI models in ONNX format, demonstrating AI-VMM's versatility across different AI workloads.

## 📦 Available Models

### Image Classification

#### MobileNetV2 (14 MB)
- **Task**: Image classification (1000 ImageNet classes)
- **Input**: 224x224 RGB image
- **Output**: Class probabilities
- **Use Case**: Lightweight, mobile-optimized classification
- **Performance**: CPU ~194ms, GPU ~198ms (model too small to show GPU benefit)

#### ResNet50 (98 MB)
- **Task**: Image classification (1000 ImageNet classes)
- **Input**: 224x224 RGB image
- **Output**: Class probabilities  
- **Use Case**: High-accuracy classification, deeper network
- **Performance**: Expected 2-3x GPU speedup vs CPU
- **Source**: [ONNX Model Zoo](https://github.com/onnx/models/tree/main/validated/vision/classification/resnet)

### Object Detection

#### YOLOv8n (13 MB)
- **Task**: Real-time object detection (80 COCO classes)
- **Input**: 640x640 RGB image
- **Output**: Bounding boxes + class probabilities
- **Use Case**: Fast real-time detection (nano variant)
- **Performance**: CPU 45ms, GPU 28ms (**1.6x speedup**)
- **Features**: Custom NMS post-processing, bounding box visualization
- **Source**: Ultralytics YOLOv8

#### SSD MobileNet (28 MB)
- **Task**: Object detection (91 COCO classes)
- **Input**: 300x300 or 1200x1200 RGB image
- **Output**: Bounding boxes + class scores
- **Use Case**: Alternative detection architecture
- **Source**: [ONNX Model Zoo](https://github.com/onnx/models/tree/main/validated/vision/object_detection_segmentation/ssd-mobilenetv1)

## 🎯 Model Types Demonstrated

AI-VMM supports **all** model types, not just LLMs:

| Category | Models | Status | GPU Benefit |
|----------|--------|--------|-------------|
| **Image Classification** | MobileNetV2, ResNet50 | ✅ Working | Medium-High |
| **Object Detection** | YOLOv8n, SSD MobileNet | ✅ Working | Medium |
| **Image Segmentation** | DeepLabV3, U-Net | 🔄 Planned | High |
| **Text Generation** | TinyLlama, GPT-2 | 🔄 Planned | Very High |
| **Multi-Modal** | CLIP, BLIP | 🔄 Planned | High |
| **Speech** | Whisper, Wav2Vec2 | 🔄 Planned | High |
| **Embeddings** | BERT, RoBERTa | 🔄 Planned | Medium |

## 📥 Adding New Models

### From ONNX Model Zoo
```bash
cd /root/everplan.ai-vmm/models
wget https://github.com/onnx/models/raw/main/.../model.onnx
```

### From HuggingFace (using Optimum)
```python
from optimum.onnxruntime import ORTModelForSequenceClassification

model = ORTModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased-finetuned-sst-2-english",
    export=True
)
model.save_pretrained("./distilbert-onnx")
```

### From PyTorch (manual export)
```python
import torch

model = YourModel()
dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    opset_version=14,
    input_names=['input'],
    output_names=['output']
)
```

## 🔧 Model Requirements

- **Format**: ONNX (.onnx files)
- **Opset**: 11+ (14 recommended)
- **Quantization**: FP32, FP16, INT8 supported
- **Dynamic shapes**: Supported for batch dimension

## 📊 Performance Testing

Test any model with the benchmark tool:

```bash
cd /root/everplan.ai-vmm/build
./examples/performance_comparison/ai_vmm_performance_comparison \
    --model ../models/resnet50.onnx \
    --device all \
    --iterations 100
```

## 🌐 Web Interface

Access models via the web dashboard:
- http://localhost:8000

Models are automatically detected from this directory and shown in the UI.

## 🎯 Demonstrating VMM Value

**vs Ollama** (LLM-only runtime):
- ✅ Supports ALL model types, not just text generation
- ✅ Heterogeneous hardware (CPU + GPU + NPU)
- ✅ Multi-modal AI workloads

**vs TensorRT** (NVIDIA-only):
- ✅ Vendor-agnostic (Intel, NVIDIA, AMD)
- ✅ Single API for all accelerators
- ✅ Dynamic device selection

**vs OpenVINO directly**:
- ✅ Abstraction layer - simpler API
- ✅ Multi-backend support
- ✅ Automatic optimization

## 📚 Resources

- [ONNX Model Zoo](https://github.com/onnx/models)
- [HuggingFace Optimum](https://huggingface.co/docs/optimum)
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [OpenVINO Model Zoo](https://github.com/openvinotoolkit/open_model_zoo)
