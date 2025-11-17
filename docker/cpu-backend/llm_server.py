#!/usr/bin/env python3
"""
OpenVINO CPU LLM Inference Server
Uses ONNX Runtime with OpenVINO Execution Provider
Based on working src/backends/llm_inference.py approach
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import uvicorn
import time
import os
import sys
import onnxruntime as ort
import numpy as np
from transformers import AutoTokenizer

app = FastAPI(title="OpenVINO CPU Backend", version="1.0.0")

# Global model cache
_session = None
_tokenizer = None
_model_path = "/app/models/tinyllama_110m_quantized.onnx"
_tokenizer_path = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    messages: List[ChatMessage]
    model: str = "tinyllama"
    max_tokens: int = 150
    temperature: float = 0.7
    stream: bool = False

def load_model():
    """Load model on first request (lazy loading)"""
    global _session, _tokenizer
    
    if _session is not None:
        return _session, _tokenizer
    
    print(f"Loading model: {_model_path}")
    print(f"Using ONNX Runtime with OpenVINO CPU Execution Provider")
    start = time.time()
    
    # Load tokenizer
    _tokenizer = AutoTokenizer.from_pretrained(_tokenizer_path, trust_remote_code=True)
    
    # Create ONNX Runtime session with OpenVINO EP
    sess_options = ort.SessionOptions()
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    
    providers = [
        ('OpenVINOExecutionProvider', {
            'device_type': 'CPU',
            'enable_dynamic_shapes': True,
            'precision': 'FP32'
        }),
        'CPUExecutionProvider'
    ]
    
    _session = ort.InferenceSession(
        _model_path,
        sess_options=sess_options,
        providers=providers
    )
    
    print(f"Model loaded in {time.time()-start:.2f}s")
    print(f"Active providers: {_session.get_providers()}")
    return _session, _tokenizer

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "backend": "ONNX Runtime + OpenVINO CPU"}

@app.get("/v1/models")
async def list_models():
    """List available models"""
    return {
        "object": "list",
        "data": [{
            "id": "tinyllama-cpu",
            "object": "model",
            "created": int(time.time()),
            "owned_by": "onnxruntime",
            "root": _model_path,
            "parent": None
        }]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """
    OpenAI-compatible chat completions endpoint (CPU backend)
    """
    start_time = time.time()
    
    try:
        session, tokenizer = load_model()
        
        # Format messages into prompt
        if request.messages:
            prompt = ""
            for msg in request.messages:
                if msg.role == "system":
                    prompt += f"System: {msg.content}\n"
                elif msg.role == "user":
                    prompt += f"User: {msg.content}\n"
                elif msg.role == "assistant":
                    prompt += f"Assistant: {msg.content}\n"
            prompt += "Assistant:"
        else:
            raise HTTPException(status_code=400, detail="No messages provided")
        
        # Tokenize input
        inputs = tokenizer(prompt, return_tensors="np", padding=True)
        input_ids = inputs["input_ids"].astype(np.int64)
        attention_mask = inputs["attention_mask"].astype(np.int64)
        
        # Prepare model inputs
        ort_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
        
        # Run inference
        time_to_first_token = time.time()
        outputs = session.run(None, ort_inputs)
        time_to_first_token = (time.time() - time_to_first_token) * 1000
        
        # Get logits and decode
        logits = outputs[0]
        
        # Simple greedy decoding (for now)
        next_token_id = np.argmax(logits[0, -1, :])
        generated_tokens = [next_token_id]
        
        # Generate more tokens
        for _ in range(min(request.max_tokens - 1, 50)):
            # Append new token
            new_input_ids = np.concatenate([input_ids, [[next_token_id]]], axis=1)
            new_attention_mask = np.ones_like(new_input_ids)
            
            ort_inputs = {
                "input_ids": new_input_ids,
                "attention_mask": new_attention_mask
            }
            
            outputs = session.run(None, ort_inputs)
            logits = outputs[0]
            next_token_id = np.argmax(logits[0, -1, :])
            
            # Check for EOS
            if next_token_id == tokenizer.eos_token_id:
                break
                
            generated_tokens.append(next_token_id)
            input_ids = new_input_ids
        
        # Decode generated text
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        # Calculate metrics
        total_time = time.time() - start_time
        tokens_generated = len(generated_tokens)
        tokens_per_sec = tokens_generated / total_time if total_time > 0 else 0
        
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": generated_text
                },
                "finish_reason": "stop"
            }],
            "usage": {
                "prompt_tokens": len(input_ids[0]),
                "completion_tokens": tokens_generated,
                "total_tokens": len(input_ids[0]) + tokens_generated
            },
            "metrics": {
                "total_time_sec": round(total_time, 3),
                "tokens_per_sec": round(tokens_per_sec, 2),
                "time_to_first_token_ms": round(time_to_first_token, 2),
                "providers": session.get_providers()
            }
        }
        
    except Exception as e:
        import traceback
        print(f"ERROR: {str(e)}", file=sys.stderr)
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")

if __name__ == "__main__":
    print(f"Starting ONNX Runtime + OpenVINO CPU Backend Server")
    print(f"Model: {_model_path}")
    print(f"Tokenizer: {_tokenizer_path}")
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8002,
        log_level="info"
    )
