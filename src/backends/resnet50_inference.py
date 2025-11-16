#!/usr/bin/env python3
"""
ResNet50 Image Classification Inference
Uses ONNX Runtime with OpenVINO backend for Intel GPU acceleration
"""

import onnxruntime as ort
import numpy as np
from PIL import Image
import json
import time
import os

def load_imagenet_classes():
    """Load full ImageNet class labels from JSON file"""
    # Try to load from models directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    classes_path = os.path.join(project_root, 'models', 'imagenet_classes.json')
    
    if os.path.exists(classes_path):
        with open(classes_path, 'r') as f:
            return json.load(f)
    else:
        # Fallback to basic classes
        return [f"class_{i}" for i in range(1000)]

def preprocess_image(image_path):
    """
    Preprocess image for ResNet50:
    - Resize to 224x224
    - Convert to RGB
    - Normalize with ImageNet mean/std
    - CHW format (channels first)
    """
    # Load and resize image
    img = Image.open(image_path).convert('RGB')
    img = img.resize((224, 224), Image.BILINEAR)
    
    # Convert to numpy array and normalize
    img_data = np.array(img).astype(np.float32)
    
    # ImageNet normalization: mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32) * 255.0
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32) * 255.0
    
    img_data = (img_data - mean) / std
    img_data = img_data.astype(np.float32)
    
    # Transpose to CHW format
    img_data = np.transpose(img_data, (2, 0, 1))
    
    # Add batch dimension
    img_data = np.expand_dims(img_data, axis=0)
    
    return img_data

def run_inference(model_path, image_path, device='CPU'):
    """
    Run ResNet50 inference on image
    
    Args:
        model_path: Path to ResNet50 ONNX model
        image_path: Path to input image
        device: 'CPU' or 'GPU'
        
    Returns:
        dict with predictions and metadata
    """
    # Set up ONNX Runtime session
    if device.upper() == 'GPU':
        providers = ['OpenVINOExecutionProvider', 'CPUExecutionProvider']
    else:
        providers = ['CPUExecutionProvider']
    
    # Create session
    session = ort.InferenceSession(model_path, providers=providers)
    
    # Get input name
    input_name = session.get_inputs()[0].name
    
    # Preprocess image
    img_data = preprocess_image(image_path)
    
    # Run inference with timing
    start_time = time.time()
    outputs = session.run(None, {input_name: img_data})
    inference_time = (time.time() - start_time) * 1000  # Convert to ms
    
    # Get predictions
    predictions = outputs[0][0]  # Remove batch dimension
    
    # Get top 5 predictions
    top5_indices = np.argsort(predictions)[-5:][::-1]
    
    # Load class labels
    classes = load_imagenet_classes()
    
    # Format results
    results = []
    for idx in top5_indices:
        score = float(predictions[idx])
        # Apply softmax to get probability
        prob = np.exp(score) / np.sum(np.exp(predictions))
        
        class_name = classes[int(idx)] if isinstance(classes, list) else classes.get(int(idx), f"class_{idx}")
        results.append({
            "class": class_name,
            "class_id": int(idx),
            "confidence": float(prob),
            "score": float(score)
        })
    
    return {
        "predictions": results,
        "inference_time_ms": inference_time,
        "providers": [p[0] if isinstance(p, tuple) else p for p in session.get_providers()],
        "device": device
    }

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Usage: python resnet50_inference.py <model_path> <image_path> [device]")
        sys.exit(1)
    
    model_path = sys.argv[1]
    image_path = sys.argv[2]
    device = sys.argv[3] if len(sys.argv) > 3 else 'CPU'
    
    result = run_inference(model_path, image_path, device)
    print(json.dumps(result, indent=2))
