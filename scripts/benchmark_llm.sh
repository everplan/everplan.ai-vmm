#!/bin/bash
# Benchmark LLM inference on CPU and GPU

set -e

cd /root/everplan.ai-vmm
source web/venv/bin/activate

MODEL_PATH="models/tinyllama_openvino"
PROMPTS=(
    "The capital of France is"
    "Once upon a time"
    "Explain quantum computing in simple terms:"
    "Write a haiku about AI"
)

echo "========================================="
echo "LLM Benchmark - TinyLlama-1.1B"
echo "========================================="
echo ""

# CPU Benchmark
echo "Testing on CPU (Intel Xeon w7-3455)..."
echo "--------------------------------------"

for i in "${!PROMPTS[@]}"; do
    prompt="${PROMPTS[$i]}"
    echo ""
    echo "[$((i+1))/${#PROMPTS[@]}] Prompt: \"$prompt\""
    
    python3 src/backends/llm_openvino.py \
        --model "$MODEL_PATH" \
        --prompt "$prompt" \
        --max-tokens 30 \
        --temperature 0.7 \
        --device CPU 2>&1 | tee /tmp/llm_bench_cpu_$i.log | grep -E "tokens|✓|text"
done

echo ""
echo "CPU Summary:"
echo "------------"
grep '"tokens_per_sec":' /tmp/llm_bench_cpu_*.log | grep -o '[0-9.]*,' | tr -d ',' | awk '{
    sum += $1; count++
} END {
    printf "Average: %.2f tokens/sec\n", sum/count
    printf "Total runs: %d\n", count
    printf "Min-Max: %.2f - %.2f tokens/sec\n", min, max
}'

# GPU Benchmark (if available)
echo ""
echo ""
echo "Testing on GPU (Intel Arc B580)..."
echo "-----------------------------------"

# Check if GPU is available first
python3 -c "
from openvino import Core
core = Core()
devices = core.available_devices
if 'GPU' in devices:
    print('GPU device found')
    exit(0)
else:
    print('GPU not available to OpenVINO')
    print('Available devices:', devices)
    exit(1)
" 2>&1

if [ $? -eq 0 ]; then
    # GPU is available
    for i in "${!PROMPTS[@]}"; do
        prompt="${PROMPTS[$i]}"
        echo ""
        echo "[$((i+1))/${#PROMPTS[@]}] Prompt: \"$prompt\""
        
        python3 src/backends/llm_openvino.py \
            --model "$MODEL_PATH" \
            --prompt "$prompt" \
            --max-tokens 30 \
            --temperature 0.7 \
            --device GPU 2>&1 | tee /tmp/llm_bench_gpu_$i.log | grep -E "tokens|✓|text"
    done
    
    echo ""
    echo "GPU Summary:"
    echo "------------"
    grep "tokens/sec" /tmp/llm_bench_gpu_*.log | awk '{
        sum += $NF; count++
    } END {
        printf "Average: %.2f tokens/sec\n", sum/count
        printf "Runs: %d\n", count
    }'
    
    # Speedup calculation
    echo ""
    echo "CPU vs GPU Comparison:"
    echo "----------------------"
    cpu_avg=$(grep "tokens/sec" /tmp/llm_bench_cpu_*.log | awk '{sum += $NF; count++} END {print sum/count}')
    gpu_avg=$(grep "tokens/sec" /tmp/llm_bench_gpu_*.log | awk '{sum += $NF; count++} END {print sum/count}')
    echo "CPU Average: ${cpu_avg} tokens/sec"
    echo "GPU Average: ${gpu_avg} tokens/sec"
    speedup=$(echo "scale=2; $gpu_avg / $cpu_avg" | bc)
    echo "GPU Speedup: ${speedup}x"
else
    echo ""
    echo "Skipping GPU benchmarks (device not available)"
    echo "Note: GPU drivers (intel-level-zero-gpu) may need to be installed"
fi

echo ""
echo "========================================="
echo "Benchmark Complete!"
echo "========================================="
