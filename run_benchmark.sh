#!/bin/bash
# Clean performance comparison runner

cd /root/everplan.ai-vmm/build/examples/performance_comparison

# Run and filter output to show only benchmark results
./ai_vmm_performance_comparison 2>/dev/null | grep -E "^(╔|║|╚|📊|---|•|====|Benchmarking|Iterations|Average|Min|Max|Throughput|📈|Model|Framework|Benchmark|Version)"
