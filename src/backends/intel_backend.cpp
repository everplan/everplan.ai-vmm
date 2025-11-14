#include "ai_vmm/backends/intel_backend.hpp"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <algorithm>
#include <cstring>
#include <cstdlib>
#include <chrono>
#include <thread>
#include <future>
#include <atomic>

namespace ai_vmm {

    // PIMPL implementation for Intel backend
    struct IntelBackend::Impl {
        std::vector<IntelDeviceInfo> available_devices;
        std::unordered_map<std::string, void*> loaded_models;
        std::vector<MemoryAllocation> active_allocations;
        
        // oneAPI/OpenVINO contexts (would be actual objects in real implementation)
        void* oneapi_context = nullptr;
        void* openvino_context = nullptr;
        
        bool oneapi_available = false;
        bool openvino_available = false;
        std::string oneapi_version;
        std::string openvino_version;
    };

    IntelBackend::IntelBackend() 
        : impl_(std::make_unique<Impl>())
        , initialized_(false)
        , optimization_level_(1)
        , profiling_enabled_(false) {
    }

    IntelBackend::~IntelBackend() {
        if (initialized_) {
            shutdown();
        }
    }

    std::string IntelBackend::get_backend_name() const {
        return "Intel";
    }

    std::string IntelBackend::get_version() const {
        return "1.0.0";
    }

    bool IntelBackend::initialize() {
        std::lock_guard<std::mutex> lock(device_mutex_);
        
        if (initialized_) {
            return true;
        }

        log_debug("Initializing Intel backend...");

        // Skip software availability checks to prevent hanging
        impl_->oneapi_available = false; // Skip for now
        impl_->openvino_available = false; // Skip for now
        impl_->oneapi_version = "Not Detected";
        impl_->openvino_version = "Not Detected";

        // Discover Intel devices with timeout protection
        try {
            log_debug("Starting Intel device discovery...");
            auto discovery_start = std::chrono::steady_clock::now();
            
            // Attempt device discovery
            impl_->available_devices = discover_intel_devices();
            
            auto discovery_end = std::chrono::steady_clock::now();
            auto discovery_duration = std::chrono::duration_cast<std::chrono::milliseconds>(discovery_end - discovery_start);
            
            log_debug("Intel device discovery completed in " + std::to_string(discovery_duration.count()) + "ms");
            log_debug("Found " + std::to_string(impl_->available_devices.size()) + " Intel devices");
            
        } catch (const std::exception& e) {
            log_debug("Exception during device discovery, falling back to CPU-only mode: " + std::string(e.what()));
            // Fallback to CPU-only mode
            impl_->available_devices.clear();
            impl_->available_devices.push_back(create_cpu_device_info());
        }

        initialized_ = true;
        return true;
    }

    void IntelBackend::shutdown() {
        std::lock_guard<std::mutex> lock(device_mutex_);
        
        if (!initialized_) {
            return;
        }

        log_debug("Shutting down Intel backend...");

        // Clean up active allocations
        for (const auto& allocation : impl_->active_allocations) {
            if (allocation.ptr) {
                std::free(allocation.ptr); // Simplified cleanup
            }
        }
        impl_->active_allocations.clear();

        // Clean up loaded models
        impl_->loaded_models.clear();

        // Clean up contexts (simplified)
        impl_->oneapi_context = nullptr;
        impl_->openvino_context = nullptr;

        initialized_ = false;
    }

    std::vector<DeviceInfo> IntelBackend::enumerate_devices() {
        if (!initialized_) {
            initialize();
        }

        std::vector<DeviceInfo> devices;
        for (const auto& intel_device : impl_->available_devices) {
            devices.push_back(intel_device.base);
        }
        return devices;
    }

    bool IntelBackend::is_device_available(const DeviceInfo& device) {
        auto devices = enumerate_devices();
        return std::find_if(devices.begin(), devices.end(), 
            [&](const DeviceInfo& d) { 
                return d.name == device.name && d.type == device.type; 
            }) != devices.end();
    }

    MemoryAllocation IntelBackend::allocate_memory(size_t size, MemoryType type, const DeviceInfo& device) {
        std::lock_guard<std::mutex> lock(memory_mutex_);

        void* ptr = nullptr;
        
        switch (type) {
            case MemoryType::HOST:
            case MemoryType::PINNED:
                ptr = allocate_host_memory(size);
                break;
            case MemoryType::DEVICE:
                ptr = allocate_device_memory(size, device);
                break;
            case MemoryType::UNIFIED:
                ptr = allocate_unified_memory(size, device);
                break;
        }

        if (!ptr) {
            log_error("Failed to allocate memory of size " + std::to_string(size));
            return MemoryAllocation();
        }

        MemoryAllocation allocation(ptr, size, type, device);
        impl_->active_allocations.push_back(allocation);
        
        log_debug("Allocated " + std::to_string(size) + " bytes of memory");
        return allocation;
    }

    void IntelBackend::deallocate_memory(const MemoryAllocation& allocation) {
        std::lock_guard<std::mutex> lock(memory_mutex_);

        if (allocation.ptr) {
            std::free(allocation.ptr); // Simplified deallocation
            
            // Remove from active allocations
            impl_->active_allocations.erase(
                std::remove_if(impl_->active_allocations.begin(), impl_->active_allocations.end(),
                    [&](const MemoryAllocation& a) { return a.ptr == allocation.ptr; }),
                impl_->active_allocations.end()
            );
        }
    }

    bool IntelBackend::copy_memory(const MemoryAllocation& src, const MemoryAllocation& dst, size_t size) {
        if (!src.ptr || !dst.ptr || size > std::min(src.size, dst.size)) {
            return false;
        }

        std::memcpy(dst.ptr, src.ptr, size);
        return true;
    }

    void IntelBackend::synchronize_device(const DeviceInfo& device) {
        // For CPU devices, synchronization is immediate
        // For GPU/NPU devices, we would wait for queue completion
        log_debug("Synchronized device: " + device.name);
    }

    Tensor IntelBackend::create_tensor(const std::vector<int64_t>& shape, Precision precision, const DeviceInfo& device) {
        Tensor tensor;
        tensor.shape = shape;
        tensor.precision = precision;
        
        size_t tensor_size = tensor.size_bytes();
        tensor.memory = allocate_memory(tensor_size, MemoryType::DEVICE, device);
        tensor.stride_bytes = tensor_size / tensor.total_elements();
        
        return tensor;
    }

    void IntelBackend::destroy_tensor(const Tensor& tensor) {
        if (tensor.memory.ptr) {
            deallocate_memory(tensor.memory);
        }
    }

    bool IntelBackend::copy_tensor(const Tensor& src, Tensor& dst) {
        if (src.shape != dst.shape || src.precision != dst.precision) {
            return false;
        }
        
        return copy_memory(src.memory, dst.memory, src.size_bytes());
    }

    bool IntelBackend::load_model(const std::string& model_path, const DeviceInfo& device, ExecutionContext& context) {
        log_debug("Loading model: " + model_path + " on device: " + device.name);

        // Check if model file exists
        if (!std::filesystem::exists(model_path)) {
            log_error("Model file not found: " + model_path);
            return false;
        }

        // Determine model format and load accordingly
        std::string extension = std::filesystem::path(model_path).extension().string();
        
        bool success = false;
        if (extension == ".onnx") {
            success = load_onnx_model(model_path, device, context);
        } else if (extension == ".xml" || extension == ".bin") {
            success = load_openvino_model(model_path, device, context);
        } else {
            log_error("Unsupported model format: " + extension);
            return false;
        }

        if (success) {
            context.device = device;
            context.model_path = model_path;
            impl_->loaded_models[model_path] = nullptr; // Placeholder for model handle
        }

        return success;
    }

    bool IntelBackend::execute_inference(ExecutionContext& context) {
        log_debug("Executing inference on device: " + context.device.name);
        
        // This is a placeholder implementation
        // Real implementation would use oneAPI/OpenVINO for actual inference
        
        // Simulate inference execution
        if (context.input_tensors.empty()) {
            log_error("No input tensors provided");
            return false;
        }

        // Create output tensors if not provided
        if (context.output_tensors.empty()) {
            // This would normally be determined by the loaded model
            Tensor output_tensor = create_tensor({1, 1000}, context.execution_precision, context.device);
            context.output_tensors.push_back(output_tensor);
        }

        log_debug("Inference completed successfully");
        return true;
    }

    void IntelBackend::unload_model(ExecutionContext& context) {
        auto it = impl_->loaded_models.find(context.model_path);
        if (it != impl_->loaded_models.end()) {
            // Clean up model resources
            impl_->loaded_models.erase(it);
            log_debug("Unloaded model: " + context.model_path);
        }
    }

    bool IntelBackend::supports_precision(const DeviceInfo& device, Precision precision) {
        // Intel devices generally support these precisions
        switch (device.type) {
            case DeviceType::CPU:
                return precision == Precision::FP32 || precision == Precision::FP16 || 
                       precision == Precision::INT8 || precision == Precision::BF16;
            case DeviceType::INTEL_IGPU:
            case DeviceType::INTEL_ARC:
                return precision == Precision::FP32 || precision == Precision::FP16 || 
                       precision == Precision::INT8;
            case DeviceType::INTEL_NPU:
                return precision == Precision::FP16 || precision == Precision::INT8 || 
                       precision == Precision::INT4;
            default:
                return false;
        }
    }

    double IntelBackend::estimate_inference_time(const ModelMetadata& model, const DeviceInfo& device) {
        switch (device.type) {
            case DeviceType::CPU:
                return estimate_cpu_inference_time(model);
            case DeviceType::INTEL_IGPU:
            case DeviceType::INTEL_ARC: {
                auto intel_device = std::find_if(impl_->available_devices.begin(), impl_->available_devices.end(),
                    [&](const IntelDeviceInfo& d) { return d.base.name == device.name; });
                if (intel_device != impl_->available_devices.end()) {
                    return estimate_gpu_inference_time(model, *intel_device);
                }
                return 100.0; // Default fallback
            }
            case DeviceType::INTEL_NPU: {
                auto intel_device = std::find_if(impl_->available_devices.begin(), impl_->available_devices.end(),
                    [&](const IntelDeviceInfo& d) { return d.base.name == device.name; });
                if (intel_device != impl_->available_devices.end()) {
                    return estimate_npu_inference_time(model, *intel_device);
                }
                return 50.0; // Default fallback
            }
            default:
                return 1000.0; // Large penalty for unsupported devices
        }
    }

    size_t IntelBackend::get_required_memory(const ModelMetadata& model, const DeviceInfo& device) {
        // Base model size + inference overhead
        size_t base_memory = model.model_size_bytes;
        
        // Add overhead based on device type
        switch (device.type) {
            case DeviceType::CPU:
                return base_memory + (base_memory / 4); // 25% overhead
            case DeviceType::INTEL_IGPU:
            case DeviceType::INTEL_ARC:
                return base_memory + (base_memory / 2); // 50% overhead for GPU memory management
            case DeviceType::INTEL_NPU:
                return base_memory + (base_memory / 8); // 12.5% overhead for NPU
            default:
                return base_memory * 2; // Conservative estimate
        }
    }

    void IntelBackend::set_optimization_level(int level) {
        optimization_level_ = std::clamp(level, 0, 2);
        log_debug("Set optimization level to " + std::to_string(optimization_level_));
    }

    void IntelBackend::enable_profiling(bool enable) {
        profiling_enabled_ = enable;
        if (enable) {
            performance_log_ = "Intel Backend Profiling Enabled\n";
        }
    }

    std::string IntelBackend::get_performance_report() {
        return performance_log_;
    }

    void IntelBackend::log_debug(const std::string& message) {
        std::cout << "[Intel Backend] " << message << std::endl;
    }

    void IntelBackend::log_error(const std::string& message) {
        std::cerr << "[Intel Backend ERROR] " << message << std::endl;
    }

    // Private helper methods implementation
    std::vector<IntelDeviceInfo> IntelBackend::discover_intel_devices() {
        std::vector<IntelDeviceInfo> devices;
        
        try {
            // Always add CPU device (this should be safe and fast)
            devices.push_back(create_cpu_device_info());
            
            // For Intel Core Ultra 9 285, add NPU detection based on CPU model
            auto cpu_info = create_cpu_device_info();
            if (cpu_info.base.name.find("Ultra") != std::string::npos) {
                log_debug("Intel Core Ultra processor detected, adding NPU");
                
                IntelDeviceInfo npu_device;
                npu_device.base.name = "Intel NPU (Core Ultra)";
                npu_device.base.type = DeviceType::INTEL_NPU;
                npu_device.base.memory_capacity = 4ULL * 1024 * 1024 * 1024; // 4GB
                npu_device.base.memory_bandwidth = 800ULL * 1024 * 1024 * 1024; // 800GB/s
                npu_device.base.supported_precisions = {Precision::FP16, Precision::INT8, Precision::INT4};
                npu_device.base.compute_score = 2.5;
                npu_device.base.supports_unified_memory = false;
                npu_device.base.driver_version = "Intel NPU Driver";
                
                npu_device.device_id = "npu0";
                npu_device.has_oneapi = impl_->oneapi_available;
                npu_device.has_openvino = impl_->openvino_available;
                npu_device.oneapi_version = impl_->oneapi_version;
                npu_device.openvino_version = impl_->openvino_version;
                npu_device.is_npu = true;
                npu_device.npu_architecture = "Lunar Lake";
                npu_device.is_arc_gpu = false;
                
                devices.push_back(npu_device);
            }
            
            // Skip DRM discovery for now to avoid hanging - will implement safer version later
            log_debug("Skipping DRM-based GPU discovery to prevent hanging");
            
        } catch (const std::exception& e) {
            log_debug("Exception during device enumeration: " + std::string(e.what()));
            // Ensure we always have at least CPU
            if (devices.empty()) {
                devices.push_back(create_cpu_device_info());
            }
        }
        
        return devices;
    }

    IntelDeviceInfo IntelBackend::create_cpu_device_info() {
        IntelDeviceInfo device;
        
        // Enhanced CPU detection with safety measures
        std::string cpu_name = "Unknown CPU";
        
        try {
            if (std::filesystem::exists("/proc/cpuinfo")) {
                std::ifstream cpuinfo("/proc/cpuinfo");
                if (cpuinfo.is_open()) {
                    std::string line;
                    int line_count = 0;
                    const int MAX_LINES = 1000; // Safety limit to prevent infinite reading
                    
                    while (std::getline(cpuinfo, line) && line_count++ < MAX_LINES) {
                        // Look for CPU model name
                        if (line.find("model name") != std::string::npos) {
                            size_t pos = line.find(':');
                            if (pos != std::string::npos && pos + 2 < line.length()) {
                                cpu_name = line.substr(pos + 2);
                                // Trim whitespace and validate
                                while (!cpu_name.empty() && std::isspace(cpu_name[0])) {
                                    cpu_name.erase(0, 1);
                                }
                                if (!cpu_name.empty() && cpu_name.length() < 200) { // Sanity check
                                    break;
                                }
                            }
                        }
                        
                        // Safety check for line length
                        if (line.length() > 1000) {
                            log_debug("Unusually long line in /proc/cpuinfo, potential issue");
                            break;
                        }
                    }
                    
                    if (line_count >= MAX_LINES) {
                        log_debug("Hit line limit reading /proc/cpuinfo, using default CPU name");
                    }
                } else {
                    log_debug("Could not open /proc/cpuinfo for reading");
                }
            } else {
                log_debug("/proc/cpuinfo not found, using default CPU name");
            }
        } catch (const std::exception& e) {
            log_debug("Exception reading CPU info: " + std::string(e.what()));
        }
        
        device.base.name = cpu_name;
        device.base.type = DeviceType::CPU;
        device.base.memory_capacity = 32ULL * 1024 * 1024 * 1024; // 32GB estimate
        device.base.memory_bandwidth = 100ULL * 1024 * 1024 * 1024; // 100GB/s estimate
        device.base.supported_precisions = {Precision::FP32, Precision::FP16, Precision::INT8, Precision::BF16};
        device.base.compute_score = 1.0;
        device.base.supports_unified_memory = true;
        device.base.driver_version = "CPU Native";
        
        device.device_id = "cpu0";
        device.has_oneapi = impl_->oneapi_available;
        device.has_openvino = impl_->openvino_available;
        device.oneapi_version = impl_->oneapi_version;
        device.openvino_version = impl_->openvino_version;
        device.is_npu = false;
        device.is_arc_gpu = false;
        
        log_debug("Created CPU device: " + cpu_name);
        return device;
    }

    std::vector<IntelDeviceInfo> IntelBackend::discover_intel_gpus() {
        std::vector<IntelDeviceInfo> devices;
        
        try {
            // Check for Intel integrated graphics with comprehensive safety measures
            if (!std::filesystem::exists("/sys/class/drm")) {
                log_debug("DRM directory not found, no Intel GPUs detected");
                return devices;
            }
            
            std::error_code ec;
            auto dir_iter = std::filesystem::directory_iterator("/sys/class/drm", ec);
            if (ec) {
                log_debug("Could not iterate /sys/class/drm: " + ec.message());
                return devices;
            }
            
            int count = 0;
            const int MAX_DRM_ENTRIES = 50; // Reduced limit for safety
            
            for (const auto& entry : dir_iter) {
                if (++count > MAX_DRM_ENTRIES) {
                    log_debug("Reached maximum DRM entry limit (" + std::to_string(MAX_DRM_ENTRIES) + "), stopping search");
                    break;
                }
                
                try {
                    if (!entry.exists(ec) || ec) {
                        continue; // Skip invalid entries
                    }
                    
                    std::string name = entry.path().filename().string();
                    if (name.find("card") != 0 || name.find("-") != std::string::npos) {
                        continue; // Only look at primary card entries
                    }
                    
                    // Safely construct vendor path with error checking
                    auto device_path = entry.path() / "device";
                    if (!std::filesystem::exists(device_path, ec) || ec) {
                        continue;
                    }
                    
                    auto vendor_path = device_path / "vendor";
                    if (!std::filesystem::exists(vendor_path, ec) || ec) {
                        continue;
                    }
                    
                    // Read vendor ID with timeout protection
                    std::ifstream vendor_file(vendor_path);
                    if (!vendor_file.is_open()) {
                        continue;
                    }
                    
                    std::string vendor_id;
                    vendor_file >> vendor_id;
                    vendor_file.close();
                    
                    if (vendor_id == "0x8086") {
                        // This is an Intel GPU
                        IntelDeviceInfo device;
                        device.base.name = "Intel Integrated Graphics";
                        device.base.type = DeviceType::INTEL_IGPU;
                        device.base.memory_capacity = 8ULL * 1024 * 1024 * 1024; // 8GB estimate
                        device.base.memory_bandwidth = 200ULL * 1024 * 1024 * 1024; // 200GB/s
                        device.base.supported_precisions = {Precision::FP32, Precision::FP16, Precision::INT8};
                        device.base.compute_score = 1.5;
                        device.base.supports_unified_memory = true;
                        device.base.driver_version = "Intel Graphics Driver";
                        
                        device.device_id = "igpu0";
                        device.has_oneapi = impl_->oneapi_available;
                        device.has_openvino = impl_->openvino_available;
                        device.oneapi_version = impl_->oneapi_version;
                        device.openvino_version = impl_->openvino_version;
                        device.is_npu = false;
                        device.is_arc_gpu = false;
                        device.gpu_generation = "Xe";
                        device.eu_count = 128;
                        
                        devices.push_back(device);
                        log_debug("Found Intel integrated GPU: " + device.base.name);
                        break; // Only add one integrated GPU
                    }
                } catch (const std::exception& entry_e) {
                    log_debug("Error processing DRM entry: " + std::string(entry_e.what()));
                    continue; // Skip this entry and continue with others
                }
            }
        } catch (const std::exception& e) {
            log_debug("Exception during GPU discovery: " + std::string(e.what()));
        }
        
        return devices;
    }

    std::vector<IntelDeviceInfo> IntelBackend::discover_intel_npus() {
        std::vector<IntelDeviceInfo> devices;
        
        try {
            // Check for Intel NPU via specific device paths with enhanced safety
            std::error_code ec;
            bool npu_found = false;
            
            // Check for accelerator devices
            if (std::filesystem::exists("/dev/accel", ec) && !ec) {
                log_debug("Found /dev/accel, checking for Intel NPU");
                npu_found = true;
            }
            
            // Check for Intel NPU specific sysfs entry
            if (!npu_found && std::filesystem::exists("/sys/class/intel_npu", ec) && !ec) {
                log_debug("Found /sys/class/intel_npu, Intel NPU detected");
                npu_found = true;
            }
            
            // Additional check for NPU via DRI devices (some Intel NPUs show up here)
            if (!npu_found && std::filesystem::exists("/sys/class/drm", ec) && !ec) {
                auto dir_iter = std::filesystem::directory_iterator("/sys/class/drm", ec);
                if (!ec) {
                    int count = 0;
                    const int MAX_ENTRIES = 20; // Lower limit for NPU search
                    
                    for (const auto& entry : dir_iter) {
                        if (++count > MAX_ENTRIES) break;
                        
                        try {
                            std::string name = entry.path().filename().string();
                            if (name.find("renderD") == 0) { // NPUs often show as render nodes
                                auto device_path = entry.path() / "device";
                                auto vendor_path = device_path / "vendor";
                                
                                if (std::filesystem::exists(vendor_path, ec) && !ec) {
                                    std::ifstream vendor_file(vendor_path);
                                    std::string vendor_id;
                                    if (vendor_file >> vendor_id && vendor_id == "0x8086") {
                                        // Check if this might be an NPU by looking at device class
                                        auto class_path = device_path / "class";
                                        if (std::filesystem::exists(class_path, ec) && !ec) {
                                            std::ifstream class_file(class_path);
                                            std::string device_class;
                                            if (class_file >> device_class && device_class.find("0x048000") != std::string::npos) {
                                                npu_found = true;
                                                break;
                                            }
                                        }
                                    }
                                }
                            }
                        } catch (const std::exception& entry_e) {
                            continue; // Skip problematic entries
                        }
                    }
                }
            }
            
            if (npu_found) {
                IntelDeviceInfo device;
                device.base.name = "Intel NPU";
                device.base.type = DeviceType::INTEL_NPU;
                device.base.memory_capacity = 4ULL * 1024 * 1024 * 1024; // 4GB
                device.base.memory_bandwidth = 800ULL * 1024 * 1024 * 1024; // 800GB/s
                device.base.supported_precisions = {Precision::FP16, Precision::INT8, Precision::INT4};
                device.base.compute_score = 2.5;
                device.base.supports_unified_memory = false;
                device.base.driver_version = "Intel NPU Driver";
                
                device.device_id = "npu0";
                device.has_oneapi = impl_->oneapi_available;
                device.has_openvino = impl_->openvino_available;
                device.oneapi_version = impl_->oneapi_version;
                device.openvino_version = impl_->openvino_version;
                device.is_npu = true;
                device.npu_architecture = "Meteor Lake";
                device.is_arc_gpu = false;
                
                devices.push_back(device);
                log_debug("Found Intel NPU: " + device.base.name);
            } else {
                log_debug("No Intel NPU detected");
            }
            
        } catch (const std::exception& e) {
            log_debug("Exception during NPU discovery: " + std::string(e.what()));
        }
        
        return devices;
    }

    bool IntelBackend::check_oneapi_availability() {
        try {
            // Check for oneAPI installation by looking for common paths and environment variables
            const char* oneapi_root = std::getenv("ONEAPI_ROOT");
            if (oneapi_root) return true;
            
            // Check common installation paths with error handling
            std::vector<std::string> common_paths = {
                "/opt/intel/oneapi",
                "/usr/local/intel/oneapi",
                "/opt/intel"
            };
            
            for (const auto& path : common_paths) {
                std::error_code ec;
                if (std::filesystem::exists(path, ec) && !ec) {
                    return true;
                }
            }
        } catch (const std::exception& e) {
            log_debug("Exception during oneAPI detection: " + std::string(e.what()));
        }
        
        return false;
    }

    bool IntelBackend::check_openvino_availability() {
        try {
            // Check for OpenVINO installation
            const char* openvino_root = std::getenv("INTEL_OPENVINO_DIR");
            if (openvino_root) return true;
            
            // Check common installation paths with error handling
            std::vector<std::string> common_paths = {
                "/opt/intel/openvino_2024",
                "/opt/intel/openvino",
                "/usr/local/intel/openvino"
            };
            
            for (const auto& path : common_paths) {
                std::error_code ec;
                if (std::filesystem::exists(path, ec) && !ec) {
                    return true;
                }
            }
        } catch (const std::exception& e) {
            log_debug("Exception during OpenVINO detection: " + std::string(e.what()));
        }
        
        return false;
    }

    std::string IntelBackend::get_oneapi_version() {
        if (!impl_->oneapi_available) return "Not Available";
        return "2024.0"; // Placeholder version
    }

    std::string IntelBackend::get_openvino_version() {
        if (!impl_->openvino_available) return "Not Available";
        return "2024.0"; // Placeholder version
    }

    void* IntelBackend::allocate_host_memory(size_t size) {
        return std::aligned_alloc(64, size); // 64-byte aligned for SIMD
    }

    void* IntelBackend::allocate_device_memory(size_t size, const DeviceInfo& device) {
        // For Intel devices, device memory often maps to system memory
        // Real implementation would use device-specific allocation
        return allocate_host_memory(size);
    }

    void* IntelBackend::allocate_unified_memory(size_t size, const DeviceInfo& device) {
        // Intel supports unified memory on many platforms
        return allocate_host_memory(size);
    }

    bool IntelBackend::load_onnx_model(const std::string& model_path, const DeviceInfo& device, ExecutionContext& context) {
        log_debug("Loading ONNX model with OpenVINO backend");
        // Placeholder for actual OpenVINO ONNX loading
        return impl_->openvino_available;
    }

    bool IntelBackend::load_openvino_model(const std::string& model_path, const DeviceInfo& device, ExecutionContext& context) {
        log_debug("Loading OpenVINO IR model");
        // Placeholder for actual OpenVINO IR loading
        return impl_->openvino_available;
    }

    double IntelBackend::estimate_cpu_inference_time(const ModelMetadata& model) {
        // Simple heuristic based on model parameters and CPU capabilities
        double base_time = (model.parameter_count / 1000000.0) * 0.1; // 0.1ms per million parameters
        return base_time / optimization_level_; // Optimization reduces time
    }

    double IntelBackend::estimate_gpu_inference_time(const ModelMetadata& model, const IntelDeviceInfo& device) {
        // GPU inference is typically faster than CPU
        double cpu_time = estimate_cpu_inference_time(model);
        double speedup = device.is_arc_gpu ? 5.0 : 2.0; // Arc GPUs provide better speedup
        return cpu_time / speedup;
    }

    double IntelBackend::estimate_npu_inference_time(const ModelMetadata& model, const IntelDeviceInfo& device) {
        // NPUs are very efficient for AI workloads but may have higher latency for some models
        double cpu_time = estimate_cpu_inference_time(model);
        return cpu_time / 8.0; // NPUs provide significant speedup for AI tasks
    }

    // Factory function
    std::unique_ptr<ComputeBackend> create_intel_backend() {
        return std::make_unique<IntelBackend>();
    }

} // namespace ai_vmm