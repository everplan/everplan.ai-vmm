#include <ai_vmm/hardware_discovery.hpp>

#include <vector>
#include <memory>
#include <fstream>
#include <string>
#include <filesystem>
#include <sstream>

namespace ai_vmm {
    /**
     * @brief Discover all available hardware on the system
     * @return Vector of discovered hardware targets
     */
    std::vector<HardwareTarget> HardwareDiscovery::discover_all_hardware() {
        std::vector<HardwareTarget> hardware;
        
        // Discover CPU (always available)
        auto cpu_targets = discover_cpu_hardware();
        hardware.insert(hardware.end(), cpu_targets.begin(), cpu_targets.end());
        
        // Discover Intel hardware
        auto intel_targets = discover_intel_hardware();
        hardware.insert(hardware.end(), intel_targets.begin(), intel_targets.end());
        
        // Discover NVIDIA hardware
        auto nvidia_targets = discover_nvidia_hardware();
        hardware.insert(hardware.end(), nvidia_targets.begin(), nvidia_targets.end());
        
        // Discover AMD hardware
        auto amd_targets = discover_amd_hardware();
        hardware.insert(hardware.end(), amd_targets.begin(), amd_targets.end());
        
        return hardware;
    }
    
    std::vector<HardwareTarget> HardwareDiscovery::discover_cpu_hardware() {
        std::vector<HardwareTarget> targets;
        
        std::string cpu_name = get_cpu_info();
        HardwareTarget cpu(HardwareType::CPU, cpu_name);
        
        HardwareCapabilities caps;
        caps.fast_rnn_ops = true;  // CPUs good for sequential processing
        caps.memory_bandwidth = get_memory_bandwidth();
        caps.memory_capacity = get_system_memory();
        caps.unified_memory = true;
        caps.supported_precisions = {Precision::FP32, Precision::FP16, Precision::INT8, Precision::INT4};
        
        // Check for CPU-specific optimizations
        if (cpu_name.find("Intel") != std::string::npos) {
            caps.vendor_extensions["supports_avx512"] = has_avx512() ? "true" : "false";
            caps.vendor_extensions["supports_vnni"] = has_vnni() ? "true" : "false";
        }
        
        cpu.set_capabilities(caps);
        targets.push_back(cpu);
        
        return targets;
    }
    
    std::vector<HardwareTarget> HardwareDiscovery::discover_intel_hardware() {
        std::vector<HardwareTarget> targets;
        
        // Check for Intel integrated graphics
        if (has_intel_gpu()) {
            HardwareTarget igpu(HardwareType::INTEL_GPU, "Intel Integrated Graphics");
            HardwareCapabilities caps;
            caps.fast_conv_ops = true;
            caps.memory_bandwidth = 50000; // ~50 GB/s estimate for iGPU
            caps.memory_capacity = get_igpu_memory();
            caps.unified_memory = true;  // Shares system memory
            caps.supported_precisions = {Precision::FP32, Precision::FP16, Precision::INT8};
            igpu.set_capabilities(caps);
            targets.push_back(igpu);
        }
        
        // Check for Intel ARC discrete GPU
        if (has_intel_arc()) {
            HardwareTarget arc(HardwareType::INTEL_ARC, get_intel_arc_name());
            HardwareCapabilities caps;
            caps.fast_conv_ops = true;
            caps.fast_attention_ops = true;
            caps.memory_bandwidth = 500000; // ~500 GB/s estimate
            caps.memory_capacity = get_arc_memory();
            caps.unified_memory = false;
            caps.supported_precisions = {Precision::FP32, Precision::FP16, Precision::BF16, Precision::INT8};
            arc.set_capabilities(caps);
            targets.push_back(arc);
        }
        
        // Check for Intel NPU
        if (has_intel_npu()) {
            HardwareTarget npu(HardwareType::INTEL_NPU, get_intel_npu_name());
            HardwareCapabilities caps;
            caps.fast_attention_ops = true;
            caps.large_embedding_support = true;
            caps.memory_bandwidth = 100000; // NPU-specific bandwidth
            caps.memory_capacity = get_npu_memory();
            caps.unified_memory = false;
            caps.supported_precisions = {Precision::FP16, Precision::INT8, Precision::INT4};
            caps.vendor_extensions["npu_generation"] = get_npu_generation();
            npu.set_capabilities(caps);
            targets.push_back(npu);
        }
        
        return targets;
    }
    
    std::vector<HardwareTarget> HardwareDiscovery::discover_nvidia_hardware() {
        std::vector<HardwareTarget> targets;
        
        // Check for NVIDIA GPUs
        auto nvidia_devices = get_nvidia_devices();
        for (const auto& device : nvidia_devices) {
            HardwareTarget gpu(HardwareType::NVIDIA_GPU, device.name);
            HardwareCapabilities caps;
            caps.fast_conv_ops = true;
            caps.fast_attention_ops = true;
            caps.tensor_cores = device.has_tensor_cores;
            caps.memory_bandwidth = device.memory_bandwidth;
            caps.memory_capacity = device.memory_capacity;
            caps.unified_memory = false;
            caps.supported_precisions = get_nvidia_supported_precisions(device);
            caps.vendor_extensions["compute_capability"] = device.compute_capability;
            caps.vendor_extensions["cuda_cores"] = std::to_string(device.cuda_cores);
            gpu.set_capabilities(caps);
            targets.push_back(gpu);
        }
        
        return targets;
    }
    
    std::vector<HardwareTarget> HardwareDiscovery::discover_amd_hardware() {
        std::vector<HardwareTarget> targets;
        
        // Check for AMD GPUs via ROCm
        auto amd_devices = get_amd_devices();
        for (const auto& device : amd_devices) {
            HardwareTarget gpu(HardwareType::AMD_GPU, device.name);
            HardwareCapabilities caps;
            caps.fast_conv_ops = true;
            caps.fast_attention_ops = true;
            caps.memory_bandwidth = device.memory_bandwidth;
            caps.memory_capacity = device.memory_capacity;
            caps.unified_memory = false;
            caps.supported_precisions = {Precision::FP32, Precision::FP16, Precision::BF16, Precision::INT8};
            caps.vendor_extensions["rocm_version"] = device.rocm_version;
            gpu.set_capabilities(caps);
            targets.push_back(gpu);
        }
        
        return targets;
    }
    
    // Helper functions for hardware detection
    std::string HardwareDiscovery::get_cpu_info() {
        std::ifstream cpuinfo("/proc/cpuinfo");
        std::string line;
        while (std::getline(cpuinfo, line)) {
            if (line.find("model name") != std::string::npos) {
                size_t colon = line.find(':');
                if (colon != std::string::npos) {
                    std::string name = line.substr(colon + 1);
                    // Trim whitespace
                    name.erase(0, name.find_first_not_of(" \t"));
                    name.erase(name.find_last_not_of(" \t") + 1);
                    return name;
                }
            }
        }
        return "Unknown CPU";
    }
    
    size_t HardwareDiscovery::get_memory_bandwidth() {
        // Estimate based on DDR type - this could be improved with actual detection
        return 50000; // ~50 GB/s estimate for DDR4
    }
    
    size_t HardwareDiscovery::get_system_memory() {
        std::ifstream meminfo("/proc/meminfo");
        std::string line;
        while (std::getline(meminfo, line)) {
            if (line.find("MemTotal:") == 0) {
                std::istringstream iss(line);
                std::string label;
                size_t kb;
                iss >> label >> kb;
                return kb * 1024; // Convert KB to bytes
            }
        }
        return 8ULL * 1024 * 1024 * 1024; // 8GB default
    }
    
    bool HardwareDiscovery::has_avx512() {
        std::ifstream cpuinfo("/proc/cpuinfo");
        std::string line;
        while (std::getline(cpuinfo, line)) {
            if (line.find("flags") != std::string::npos && 
                line.find("avx512f") != std::string::npos) {
                return true;
            }
        }
        return false;
    }
    
    bool HardwareDiscovery::has_vnni() {
        std::ifstream cpuinfo("/proc/cpuinfo");
        std::string line;
        while (std::getline(cpuinfo, line)) {
            if (line.find("flags") != std::string::npos && 
                line.find("avx512_vnni") != std::string::npos) {
                return true;
            }
        }
        return false;
    }
    
    bool HardwareDiscovery::has_intel_gpu() {
        // Check for Intel GPU via /sys/class/drm
        return std::filesystem::exists("/sys/class/drm/card0") && 
               check_vendor_id("8086"); // Intel vendor ID
    }
    
    bool HardwareDiscovery::has_intel_arc() {
        // Check for Intel ARC discrete GPU
        return check_pci_device("8086", "56a0"); // Example ARC device ID
    }
    
    bool HardwareDiscovery::has_intel_npu() {
        // Check for Intel NPU - this would need actual NPU detection logic
        return std::filesystem::exists("/dev/intel_npu") ||
               check_pci_device("8086", "643e"); // Example NPU device ID
    }
    
    size_t HardwareDiscovery::get_igpu_memory() {
        // Intel iGPU shares system memory
        return get_system_memory() / 2; // Estimate half available for GPU
    }
    
    size_t HardwareDiscovery::get_arc_memory() {
        // Read ARC memory from sysfs or estimate
        return 8ULL * 1024 * 1024 * 1024; // 8GB estimate
    }
    
    size_t HardwareDiscovery::get_npu_memory() {
        // NPU typically has dedicated memory
        return 2ULL * 1024 * 1024 * 1024; // 2GB estimate
    }
    
    std::string HardwareDiscovery::get_intel_arc_name() {
        return "Intel ARC GPU"; // Could be improved with actual detection
    }
    
    std::string HardwareDiscovery::get_intel_npu_name() {
        return "Intel NPU"; // Could be improved with version detection
    }
    
    std::string HardwareDiscovery::get_npu_generation() {
        return "1.0"; // Placeholder
    }
    
    bool HardwareDiscovery::check_vendor_id(const std::string& vendor_id) {
        try {
            for (const auto& entry : std::filesystem::directory_iterator("/sys/bus/pci/devices")) {
                std::ifstream vendor_file(entry.path() / "vendor");
                std::string vendor;
                if (vendor_file >> vendor && vendor == ("0x" + vendor_id)) {
                    return true;
                }
            }
        } catch (...) {
            // Ignore filesystem errors
        }
        return false;
    }
    
    bool HardwareDiscovery::check_pci_device(const std::string& vendor_id, const std::string& device_id) {
        try {
            for (const auto& entry : std::filesystem::directory_iterator("/sys/bus/pci/devices")) {
                std::ifstream vendor_file(entry.path() / "vendor");
                std::ifstream device_file(entry.path() / "device");
                std::string vendor, device;
                if (vendor_file >> vendor && device_file >> device) {
                    if (vendor == ("0x" + vendor_id) && device == ("0x" + device_id)) {
                        return true;
                    }
                }
            }
        } catch (...) {
            // Ignore filesystem errors
        }
        return false;
    }
    
    // NVIDIA detection structures and functions
    struct NVIDIADevice {
        std::string name;
        bool has_tensor_cores;
        size_t memory_bandwidth;
        size_t memory_capacity;
        std::string compute_capability;
        int cuda_cores;
    };
    
    std::vector<HardwareDiscovery::NVIDIADevice> HardwareDiscovery::get_nvidia_devices() {
        std::vector<NVIDIADevice> devices;
        
        // Check for NVIDIA GPUs via nvidia-smi or /proc/driver/nvidia
        if (std::filesystem::exists("/proc/driver/nvidia/version")) {
            // Placeholder - real implementation would parse nvidia-smi output
            // or use CUDA runtime API if available
            NVIDIADevice gpu;
            gpu.name = "NVIDIA GPU";
            gpu.has_tensor_cores = true;
            gpu.memory_bandwidth = 900000; // ~900 GB/s estimate
            gpu.memory_capacity = 24ULL * 1024 * 1024 * 1024; // 24GB
            gpu.compute_capability = "8.9";
            gpu.cuda_cores = 10752;
            devices.push_back(gpu);
        }
        
        return devices;
    }
    
    std::vector<Precision> HardwareDiscovery::get_nvidia_supported_precisions(const NVIDIADevice& device) {
        std::vector<Precision> precisions = {Precision::FP32, Precision::FP16, Precision::INT8};
        
        if (device.has_tensor_cores) {
            precisions.push_back(Precision::BF16);
            precisions.push_back(Precision::FP8);
        }
        
        return precisions;
    }
    
    // AMD detection structures and functions  
    struct AMDDevice {
        std::string name;
        size_t memory_bandwidth;
        size_t memory_capacity;
        std::string rocm_version;
    };
    
    std::vector<HardwareDiscovery::AMDDevice> HardwareDiscovery::get_amd_devices() {
        std::vector<AMDDevice> devices;
        
        // Check for AMD GPUs via ROCm
        if (std::filesystem::exists("/opt/rocm") || 
            std::filesystem::exists("/sys/class/kfd/kfd/topology/nodes")) {
            AMDDevice gpu;
            gpu.name = "AMD GPU";
            gpu.memory_bandwidth = 1600000; // ~1.6 TB/s estimate for high-end
            gpu.memory_capacity = 64ULL * 1024 * 1024 * 1024; // 64GB
            gpu.rocm_version = "5.0";
            devices.push_back(gpu);
        }
        
        return devices;
    }
    
} // namespace ai_vmm