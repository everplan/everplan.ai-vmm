#!/usr/bin/env python3
"""
YOLOv8 Persistent Inference Server
Keeps model loaded in memory to avoid repeated initialization overhead
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


class YOLOv8InferenceServer:
    def __init__(self, model_path):
        """Initialize with model loaded for both CPU and GPU"""
        self.model_path = model_path
        self.sessions = {}
        
        # Pre-load both CPU and GPU sessions
        print(json.dumps({'status': 'loading_cpu'}), flush=True)
        self.sessions['cpu'] = ort.InferenceSession(
            model_path, 
            providers=['CPUExecutionProvider']
        )
        
        print(json.dumps({'status': 'loading_gpu'}), flush=True)
        self.sessions['gpu'] = ort.InferenceSession(
            model_path,
            providers=['OpenVINOExecutionProvider', 'CPUExecutionProvider']
        )
        
        print(json.dumps({'status': 'ready', 'providers': {
            'cpu': self.sessions['cpu'].get_providers(),
            'gpu': self.sessions['gpu'].get_providers()
        }}), flush=True)
    
    def preprocess_image(self, image_data, input_size=640):
        """Preprocess image for YOLOv8 inference"""
        img_bytes = base64.b64decode(image_data)
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
        
        orig_w, orig_h = img.size
        scale = min(input_size / orig_w, input_size / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)
        img_resized = img.resize((new_w, new_h), Image.BILINEAR)
        
        img_padded = Image.new('RGB', (input_size, input_size), (114, 114, 114))
        img_padded.paste(img_resized, (0, 0))
        
        img_np = np.array(img_padded).astype(np.float32)
        img_np = img_np / 255.0
        img_np = img_np.transpose(2, 0, 1)
        img_np = np.expand_dims(img_np, axis=0)
        
        return img_np, orig_w, orig_h, scale
    
    def postprocess_yolov8(self, output, orig_w, orig_h, scale, conf_threshold=0.25, iou_threshold=0.45):
        """Post-process YOLOv8 output to get bounding boxes"""
        if len(output.shape) == 3:
            output = output[0]
        
        predictions = output.transpose(1, 0)
        boxes = predictions[:, :4]
        scores = predictions[:, 4:]
        
        class_ids = np.argmax(scores, axis=1)
        confidences = np.max(scores, axis=1)
        
        mask = confidences > conf_threshold
        boxes = boxes[mask]
        confidences = confidences[mask]
        class_ids = class_ids[mask]
        
        if len(boxes) == 0:
            return []
        
        x_center, y_center, width, height = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        x1 = x_center - width / 2
        y1 = y_center - height / 2
        x2 = x_center + width / 2
        y2 = y_center + height / 2
        
        x1 = x1 / scale
        y1 = y1 / scale
        x2 = x2 / scale
        y2 = y2 / scale
        
        x1 = np.clip(x1, 0, orig_w)
        y1 = np.clip(y1, 0, orig_h)
        x2 = np.clip(x2, 0, orig_w)
        y2 = np.clip(y2, 0, orig_h)
        
        detections = []
        for i in range(len(boxes)):
            detections.append({
                'bbox': [float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i])],
                'class_id': int(class_ids[i]),
                'class_name': COCO_CLASSES[class_ids[i]],
                'confidence': float(confidences[i])
            })
        
        # NMS
        keep_indices = self.nms(
            np.stack([x1, y1, x2, y2], axis=1),
            confidences,
            iou_threshold
        )
        
        return [detections[i] for i in keep_indices]
    
    def nms(self, boxes, scores, iou_threshold):
        """Non-maximum suppression"""
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
    
    def infer(self, image_data, device='gpu'):
        """Run inference on image"""
        device_key = device.lower()
        if device_key not in self.sessions:
            device_key = 'cpu'
        
        session = self.sessions[device_key]
        input_name = session.get_inputs()[0].name
        
        input_tensor, orig_w, orig_h, scale = self.preprocess_image(image_data)
        output = session.run(None, {input_name: input_tensor})
        detections = self.postprocess_yolov8(output[0], orig_w, orig_h, scale)
        
        return {
            'success': True,
            'detections': detections,
            'count': len(detections),
            'providers': session.get_providers()
        }
    
    def run(self):
        """Main loop - read commands from stdin"""
        while True:
            try:
                line = sys.stdin.readline()
                if not line:
                    break
                
                request = json.loads(line.strip())
                command = request.get('command')
                
                if command == 'infer':
                    image_data = request.get('image')
                    device = request.get('device', 'gpu')
                    result = self.infer(image_data, device)
                    print(json.dumps(result), flush=True)
                
                elif command == 'ping':
                    print(json.dumps({'status': 'pong'}), flush=True)
                
                elif command == 'exit':
                    break
                
                else:
                    print(json.dumps({
                        'success': False,
                        'error': f'Unknown command: {command}'
                    }), flush=True)
            
            except Exception as e:
                print(json.dumps({
                    'success': False,
                    'error': str(e)
                }), flush=True)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print(json.dumps({
            'success': False,
            'error': 'Usage: yolov8_server.py <model_path>'
        }))
        sys.exit(1)
    
    model_path = sys.argv[1]
    server = YOLOv8InferenceServer(model_path)
    server.run()
