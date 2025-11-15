#!/usr/bin/env python3
"""
YOLOv8 Inference Script for AI-VMM
Handles object detection with proper pre/post-processing
"""

import sys
import json
import numpy as np
import onnxruntime as ort
from PIL import Image
import io
import base64

# COCO class names (80 classes)
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def preprocess_image(image_data, input_size=640):
    """Preprocess image for YOLOv8 inference"""
    # Decode base64 image
    img_bytes = base64.b64decode(image_data)
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    
    # Store original size for later
    orig_w, orig_h = img.size
    
    # Resize maintaining aspect ratio
    scale = min(input_size / orig_w, input_size / orig_h)
    new_w = int(orig_w * scale)
    new_h = int(orig_h * scale)
    img_resized = img.resize((new_w, new_h), Image.BILINEAR)
    
    # Create padded image
    img_padded = Image.new('RGB', (input_size, input_size), (114, 114, 114))
    img_padded.paste(img_resized, (0, 0))
    
    # Convert to numpy and normalize
    img_np = np.array(img_padded).astype(np.float32)
    img_np = img_np / 255.0  # Normalize to [0, 1]
    img_np = img_np.transpose(2, 0, 1)  # HWC -> CHW
    img_np = np.expand_dims(img_np, axis=0)  # Add batch dimension
    
    return img_np, orig_w, orig_h, scale


def postprocess_yolov8(output, orig_w, orig_h, scale, conf_threshold=0.25, iou_threshold=0.45):
    """Post-process YOLOv8 output to get bounding boxes"""
    # YOLOv8 output shape: [1, 84, 8400]
    # 84 = 4 (bbox) + 80 (classes)
    
    # Remove batch dimension and transpose to [8400, 84]
    if len(output.shape) == 3:
        output = output[0]  # Remove batch dim: [84, 8400]
    
    predictions = output.transpose(1, 0)  # [8400, 84]
    
    # Extract boxes and scores
    boxes = predictions[:, :4]  # [x_center, y_center, width, height]
    scores = predictions[:, 4:]  # [80 classes]
    
    # Get class with max score for each detection
    class_ids = np.argmax(scores, axis=1)
    confidences = np.max(scores, axis=1)
    
    # Filter by confidence threshold
    mask = confidences > conf_threshold
    boxes = boxes[mask]
    confidences = confidences[mask]
    class_ids = class_ids[mask]
    
    if len(boxes) == 0:
        return []
    
    # Convert from center format to corner format
    x_center, y_center, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    
    # Scale back to original image size
    x1 = x1 / scale
    y1 = y1 / scale
    x2 = x2 / scale
    y2 = y2 / scale
    
    # Clip to image boundaries
    x1 = np.clip(x1, 0, orig_w)
    y1 = np.clip(y1, 0, orig_h)
    x2 = np.clip(x2, 0, orig_w)
    y2 = np.clip(y2, 0, orig_h)
    
    # Apply NMS (Non-Maximum Suppression)
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
    indices = nms(boxes_xyxy, confidences, iou_threshold)
    
    # Format results
    results = []
    for idx in indices:
        results.append({
            'class_id': int(class_ids[idx]),
            'class_name': COCO_CLASSES[class_ids[idx]],
            'confidence': float(confidences[idx]),
            'bbox': [float(x1[idx]), float(y1[idx]), float(x2[idx]), float(y2[idx])]
        })
    
    return results


def nms(boxes, scores, iou_threshold):
    """Non-Maximum Suppression"""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return keep


def main():
    if len(sys.argv) < 4:
        print(json.dumps({
            'success': False,
            'error': 'Usage: yolov8_inference.py <model_path> <action> <device> [image_base64_or_stdin]'
        }))
        sys.exit(1)
    
    model_path = sys.argv[1]
    action = sys.argv[2]
    device = sys.argv[3]
    
    try:
        # Choose provider based on device
        if device.upper() == 'GPU':
            providers = ['OpenVINOExecutionProvider', 'CPUExecutionProvider']
        else:
            providers = ['CPUExecutionProvider']
        
        if action == 'load':
            # Just load model and return session info
            session = ort.InferenceSession(model_path, providers=providers)
            input_info = session.get_inputs()[0]
            output_info = session.get_outputs()[0]
            
            print(json.dumps({
                'success': True,
                'input': {
                    'name': input_info.name,
                    'shape': [str(d) for d in input_info.shape],
                    'type': str(input_info.type)
                },
                'output': {
                    'name': output_info.name,
                    'shape': [str(d) for d in output_info.shape],
                    'type': str(output_info.type)
                },
                'providers': session.get_providers()
            }))
        
        elif action == 'infer':
            # Read image data from stdin or argument
            if len(sys.argv) >= 5:
                image_data = sys.argv[4]
            else:
                # Read from stdin
                image_data = sys.stdin.read().strip()
            
            if not image_data:
                raise ValueError('Image data required for inference')
            
            # Load model
            session = ort.InferenceSession(model_path, providers=providers)
            input_name = session.get_inputs()[0].name
            
            # Preprocess image
            input_tensor, orig_w, orig_h, scale = preprocess_image(image_data)
            
            # Run inference
            output = session.run(None, {input_name: input_tensor})
            
            # Post-process results (output is a list with one element)
            detections = postprocess_yolov8(output[0], orig_w, orig_h, scale)
            
            print(json.dumps({
                'success': True,
                'detections': detections,
                'count': len(detections),
                'providers': session.get_providers()
            }))
        
        else:
            raise ValueError(f'Unknown action: {action}')
    
    except Exception as e:
        print(json.dumps({
            'success': False,
            'error': str(e)
        }))
        sys.exit(1)


if __name__ == '__main__':
    main()
