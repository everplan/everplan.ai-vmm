# CPU Backend - OpenVINO LLM Inference
FROM openvino/ubuntu22_runtime:2025.3.0

# Switch to root for installation
USER root

# Install Python dependencies for LLM serving
RUN apt-get update && apt-get install -y \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies for LLM serving
# Use ONNX Runtime + OpenVINO EP (simpler than optimum-intel)
RUN pip3 install --no-cache-dir \
    onnxruntime-openvino==1.17.0 \
    transformers==4.38.2 \
    fastapi==0.104.1 \
    uvicorn[standard]==0.24.0 \
    pydantic==2.5.0 \
    numpy==1.24.4

# Create working directory
WORKDIR /app

# Copy the LLM inference server script
COPY llm_server.py /app/

# Expose API port
EXPOSE 8002

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:8002/health')" || exit 1

# Run the server
CMD ["python3", "llm_server.py"]
