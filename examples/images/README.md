# Example Images

This directory contains test images for demonstrating AI-VMM inference capabilities.

## Images

### Object Detection (YOLOv8)
- **`zidane.jpg`** - Soccer players, ideal for testing person and tie detection
  - Expected detections: 2-3 persons, 1 tie
  - Resolution: 1280x720

- **`bus.jpg`** - Street scene with bus and people
  - Expected detections: bus, persons, vehicles
  - Resolution: 1280x720

## Usage

### Web Interface
1. Start the web server: `cd /root/everplan.ai-vmm/web && python3 vmm_api.py`
2. Open http://localhost:8000
3. Select model (YOLOv8n for object detection, MobileNetV2 for classification)
4. Upload an image from this folder
5. Click "Run Inference"

### Command Line (YOLOv8)
```bash
cd /root/everplan.ai-vmm
base64 examples/images/bus.jpg | python3 src/backends/yolov8_inference.py \
  models/yolov8n.onnx infer GPU | python3 -m json.tool
```

### Performance Testing
The YOLOv8 model demonstrates ~3x speedup on Intel Battlemage GPU:
- CPU: ~17ms per inference
- GPU (OpenVINO): ~5-6ms per inference

## Adding Your Own Images

Simply copy any JPG or PNG images to this directory. Images will be automatically resized to 640x640 for YOLOv8 object detection or 224x224 for MobileNetV2 classification while maintaining aspect ratio.

Recommended image characteristics:
- Format: JPG, PNG
- Resolution: 640x640 to 1920x1080 (larger images take longer to process)
- Content: Clear, well-lit objects for best detection results
- File size: < 5MB for optimal web upload performance
