#include <ai_vmm/ai_vmm.hpp>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <cstring>
#include <vector>

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]\n\n";
    std::cout << "Options:\n";
    std::cout << "  --device <type>    Force specific device type (cpu, gpu, npu, all)\n";
    std::cout << "  --iterations <n>   Number of iterations (default: 10)\n";
    std::cout << "  --help            Show this help message\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << program_name << "                    # Use VMM's automatic selection\n";
    std::cout << "  " << program_name << " --device cpu       # Force CPU execution\n";
    std::cout << "  " << program_name << " --device gpu       # Force GPU execution\n";
    std::cout << "  " << program_name << " --device all       # Benchmark all devices\n";
    std::cout << "  " << program_name << " --iterations 20    # Run 20 iterations\n";
}

struct BenchmarkResult {
    std::string device_name;
    std::string device_type;
    double avg_ms;
    double min_ms;
    double max_ms;
    double throughput;
};

BenchmarkResult run_benchmark(ai_vmm::VMM& vmm, const std::string& model_path, 
                               ai_vmm::HardwareType device_type, const std::string& device_label,
                               int iterations) {
    BenchmarkResult result;
    result.device_name = device_label;
    result.device_type = (device_type == ai_vmm::HardwareType::CPU) ? "CPU" : 
                         (device_type == ai_vmm::HardwareType::INTEL_GPU) ? "GPU" : "NPU";
    
    try {
        // Deploy model with device type preference
        ai_vmm::DeploymentConstraints constraints;
        constraints.preferred_hardware = {device_type};
        
        auto model = vmm.deploy(model_path, constraints);
        
        // Warmup run
        ai_vmm::Tensor input({1, 224, 224, 3}, ai_vmm::Precision::FP32);
        model->execute(input);
        
        // Benchmark runs
        std::vector<long long> times;
        for (int i = 0; i < iterations; i++) {
            auto start = std::chrono::high_resolution_clock::now();
            auto output = model->execute(input);
            auto end = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
            times.push_back(duration.count());
        }
        
        // Calculate statistics
        long long sum = 0;
        long long min_time = times[0];
        long long max_time = times[0];
        
        for (auto time : times) {
            sum += time;
            if (time < min_time) min_time = time;
            if (time > max_time) max_time = time;
        }
        
        double avg_time = static_cast<double>(sum) / iterations;
        
        result.avg_ms = avg_time / 1000.0;
        result.min_ms = min_time / 1000.0;
        result.max_ms = max_time / 1000.0;
        result.throughput = 1000000.0 / avg_time;
        
    } catch (const std::exception& e) {
        std::cerr << "Benchmark failed for " << device_label << ": " << e.what() << std::endl;
        result.avg_ms = -1;
    }
    
    return result;
}

void print_results(const std::vector<BenchmarkResult>& results) {
    std::cout << "\n" << std::string(80, '=') << std::endl;
    std::cout << "📊 Benchmark Results" << std::endl;
    std::cout << std::string(80, '=') << std::endl;
    
    std::cout << std::left << std::fixed << std::setprecision(2);
    std::cout << std::setw(35) << "Device" 
              << std::setw(12) << "Avg (ms)" 
              << std::setw(12) << "Min (ms)" 
              << std::setw(12) << "Max (ms)" 
              << "Throughput" << std::endl;
    std::cout << std::string(80, '-') << std::endl;
    
    for (const auto& result : results) {
        if (result.avg_ms > 0) {
            std::cout << std::setw(35) << result.device_name 
                      << std::setw(12) << result.avg_ms
                      << std::setw(12) << result.min_ms
                      << std::setw(12) << result.max_ms
                      << result.throughput << " inf/s" << std::endl;
        }
    }
    
    // Find fastest device
    double best_time = 1e9;
    std::string best_device;
    for (const auto& result : results) {
        if (result.avg_ms > 0 && result.avg_ms < best_time) {
            best_time = result.avg_ms;
            best_device = result.device_name;
        }
    }
    
    if (!best_device.empty()) {
        std::cout << std::string(80, '-') << std::endl;
        std::cout << "🏆 Fastest: " << best_device << " (" << best_time << " ms)" << std::endl;
    }
}

int main(int argc, char* argv[]) {
    try {
        // Parse command line arguments
        std::string device_filter = "auto";
        int iterations = 10;
        
        for (int i = 1; i < argc; i++) {
            if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
                print_usage(argv[0]);
                return 0;
            } else if (strcmp(argv[i], "--device") == 0 && i + 1 < argc) {
                device_filter = argv[++i];
            } else if (strcmp(argv[i], "--iterations") == 0 && i + 1 < argc) {
                iterations = std::stoi(argv[++i]);
            }
        }
        
        std::cout << "\n";
        std::cout << "╔════════════════════════════════════════════════════════════╗\n";
        std::cout << "║        AI-VMM Performance Comparison Benchmark             ║\n";
        std::cout << "╚════════════════════════════════════════════════════════════╝\n";
        std::cout << "\nVersion: " << ai_vmm::VMM::get_version() << std::endl;
        std::cout << "Iterations: " << iterations << std::endl;
        std::cout << "Device filter: " << device_filter << std::endl;
        
        // Initialize VMM
        ai_vmm::VMM vmm;
        vmm.set_debug_mode(false);
        
        // Get available hardware
        auto available_hardware = vmm.get_available_hardware();
        
        std::cout << "\n📋 Available Hardware:" << std::endl;
        std::cout << std::string(60, '-') << std::endl;
        for (const auto& hw : available_hardware) {
            std::string type_str;
            switch (hw.get_type()) {
                case ai_vmm::HardwareType::CPU: type_str = "CPU"; break;
                case ai_vmm::HardwareType::INTEL_GPU: type_str = "Intel GPU"; break;
                case ai_vmm::HardwareType::INTEL_IGPU: type_str = "Intel iGPU"; break;
                case ai_vmm::HardwareType::INTEL_ARC: type_str = "Intel Arc GPU"; break;
                case ai_vmm::HardwareType::INTEL_NPU: type_str = "Intel NPU"; break;
                case ai_vmm::HardwareType::NVIDIA_GPU: type_str = "NVIDIA GPU"; break;
                default: type_str = "Unknown"; break;
            }
            std::cout << "  • " << hw.get_name() << " [" << type_str << "]" << std::endl;
        }
        
        const std::string model_path = "models/mobilenetv2.onnx";
        std::vector<BenchmarkResult> results;
        
        std::cout << "\n⏱️  Running benchmarks..." << std::endl;
        
        // Determine which devices to benchmark
        bool test_cpu = (device_filter == "all" || device_filter == "cpu" || device_filter == "auto");
        bool test_gpu = (device_filter == "all" || device_filter == "gpu" || device_filter == "auto");
        bool test_npu = (device_filter == "all" || device_filter == "npu");
        
        // Find devices
        for (const auto& hw : available_hardware) {
            if (test_cpu && hw.get_type() == ai_vmm::HardwareType::CPU) {
                std::cout << "\nBenchmarking: " << hw.get_name() << " [CPU]..." << std::flush;
                auto result = run_benchmark(vmm, model_path, ai_vmm::HardwareType::CPU, hw.get_name(), iterations);
                if (result.avg_ms > 0) {
                    results.push_back(result);
                    std::cout << " ✓" << std::endl;
                }
                test_cpu = false; // Only test first CPU
            } else if (test_gpu && (hw.get_type() == ai_vmm::HardwareType::INTEL_GPU || 
                                     hw.get_type() == ai_vmm::HardwareType::INTEL_IGPU ||
                                     hw.get_type() == ai_vmm::HardwareType::INTEL_ARC)) {
                std::cout << "\nBenchmarking: " << hw.get_name() << " [GPU]..." << std::flush;
                auto result = run_benchmark(vmm, model_path, hw.get_type(), hw.get_name(), iterations);
                if (result.avg_ms > 0) {
                    results.push_back(result);
                    std::cout << " ✓" << std::endl;
                }
                test_gpu = false; // Only test first GPU
            } else if (test_npu && hw.get_type() == ai_vmm::HardwareType::INTEL_NPU) {
                std::cout << "\nBenchmarking: " << hw.get_name() << " [NPU]..." << std::flush;
                auto result = run_benchmark(vmm, model_path, ai_vmm::HardwareType::INTEL_NPU, hw.get_name(), iterations);
                if (result.avg_ms > 0) {
                    results.push_back(result);
                    std::cout << " ✓" << std::endl;
                }
                test_npu = false; // Only test first NPU
            }
        }
        
        // Print results
        print_results(results);
        
        std::cout << "\n" << std::string(80, '=') << std::endl;
        std::cout << "Model: MobileNetV2 (224x224x3 input)" << std::endl;
        std::cout << "Framework: ONNX Runtime with OpenVINO ExecutionProvider" << std::endl;
        std::cout << "\n✅ Benchmark completed successfully!\n" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
