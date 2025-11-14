#pragma once

#include <ai_vmm/types.hpp>

#include <vector>
#include <string>

namespace ai_vmm {

/**
 * @brief Hardware discovery and registry functionality
 */
class HardwareDiscovery {
public:
    /**
     * @brief Discover all available hardware on the system
     * @return Vector of discovered hardware targets
     */
    static std::vector<HardwareTarget> discover_all_hardware();
    
private:
    static std::vector<HardwareTarget> discover_cpu_hardware();
    static std::vector<HardwareTarget> discover_intel_hardware();
    static std::vector<HardwareTarget> discover_nvidia_hardware();
    static std::vector<HardwareTarget> discover_amd_hardware();
    
    // Helper functions for hardware detection
    static std::string get_cpu_info();
    static size_t get_memory_bandwidth();
    static size_t get_system_memory();
    static bool has_avx512();
    static bool has_vnni();
    static bool has_intel_gpu();
    static bool has_intel_arc();
    static bool has_intel_npu();
    static size_t get_igpu_memory();
    static size_t get_arc_memory();
    static size_t get_npu_memory();
    static std::string get_intel_arc_name();
    static std::string get_intel_npu_name();
    static std::string get_npu_generation();
    static bool check_vendor_id(const std::string& vendor_id);
    static bool check_pci_device(const std::string& vendor_id, const std::string& device_id);
    
    // NVIDIA detection structures and functions
    struct NVIDIADevice {
        std::string name;
        bool has_tensor_cores;
        size_t memory_bandwidth;
        size_t memory_capacity;
        std::string compute_capability;
        int cuda_cores;
    };
    
    static std::vector<NVIDIADevice> get_nvidia_devices();
    static std::vector<Precision> get_nvidia_supported_precisions(const NVIDIADevice& device);
    
    // AMD detection structures and functions  
    struct AMDDevice {
        std::string name;
        size_t memory_bandwidth;
        size_t memory_capacity;
        std::string rocm_version;
    };
    
    static std::vector<AMDDevice> get_amd_devices();
};

} // namespace ai_vmm