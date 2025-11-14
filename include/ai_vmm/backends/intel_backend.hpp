#pragma once

#include "ai_vmm/backends/compute_backend.hpp"
#include <memory>
#include <mutex>

namespace ai_vmm {

    // Intel-specific device information
    struct IntelDeviceInfo {
        DeviceInfo base;
        std::string device_id;
        bool has_oneapi;
        bool has_openvino;
        std::string oneapi_version;
        std::string openvino_version;
        
        // NPU-specific information
        bool is_npu;
        std::string npu_architecture;  // "Meteor Lake", "Arrow Lake", etc.
        
        // GPU-specific information  
        bool is_arc_gpu;
        std::string gpu_generation;    // "Xe", "Xe2", etc.
        size_t eu_count;               // Execution units
    };

    // Intel backend implementation
    class IntelBackend : public ComputeBackend {
    public:
        IntelBackend();
        virtual ~IntelBackend();

        // ComputeBackend interface implementation
        std::string get_backend_name() const override;
        std::string get_version() const override;

        // Device management
        bool initialize() override;
        void shutdown() override;
        std::vector<DeviceInfo> enumerate_devices() override;
        bool is_device_available(const DeviceInfo& device) override;

        // Memory management
        MemoryAllocation allocate_memory(size_t size, MemoryType type, const DeviceInfo& device) override;
        void deallocate_memory(const MemoryAllocation& allocation) override;
        bool copy_memory(const MemoryAllocation& src, const MemoryAllocation& dst, size_t size) override;
        void synchronize_device(const DeviceInfo& device) override;

        // Tensor operations
        Tensor create_tensor(const std::vector<int64_t>& shape, Precision precision, const DeviceInfo& device) override;
        void destroy_tensor(const Tensor& tensor) override;
        bool copy_tensor(const Tensor& src, Tensor& dst) override;

        // Model operations
        bool load_model(const std::string& model_path, const DeviceInfo& device, ExecutionContext& context) override;
        bool execute_inference(ExecutionContext& context) override;
        void unload_model(ExecutionContext& context) override;

        // Performance and capability queries
        bool supports_precision(const DeviceInfo& device, Precision precision) override;
        double estimate_inference_time(const ModelMetadata& model, const DeviceInfo& device) override;
        size_t get_required_memory(const ModelMetadata& model, const DeviceInfo& device) override;

        // Optimization hints
        void set_optimization_level(int level) override;
        void enable_profiling(bool enable) override;
        std::string get_performance_report() override;

    protected:
        void log_debug(const std::string& message) override;
        void log_error(const std::string& message) override;

    private:
        // Internal implementation details
        struct Impl;
        std::unique_ptr<Impl> impl_;
        
        // Device discovery helpers
        std::vector<IntelDeviceInfo> discover_intel_devices();
        IntelDeviceInfo create_cpu_device_info();
        std::vector<IntelDeviceInfo> discover_intel_gpus();
        std::vector<IntelDeviceInfo> discover_intel_gpus_via_pcie();
        std::string find_drm_device_for_pcie(const std::string& pcie_address);
        std::vector<IntelDeviceInfo> discover_intel_npus();
        
        // Backend capability detection
        bool check_oneapi_availability();
        bool check_openvino_availability();
        std::string get_oneapi_version();
        std::string get_openvino_version();
        
        // Memory management helpers
        void* allocate_host_memory(size_t size);
        void* allocate_device_memory(size_t size, const DeviceInfo& device);
        void* allocate_unified_memory(size_t size, const DeviceInfo& device);
        
        // Model loading helpers
        bool load_onnx_model(const std::string& model_path, const DeviceInfo& device, ExecutionContext& context);
        bool load_openvino_model(const std::string& model_path, const DeviceInfo& device, ExecutionContext& context);
        
        // Performance estimation
        double estimate_cpu_inference_time(const ModelMetadata& model);
        double estimate_gpu_inference_time(const ModelMetadata& model, const IntelDeviceInfo& device);
        double estimate_npu_inference_time(const ModelMetadata& model, const IntelDeviceInfo& device);
        
        // Thread safety
        mutable std::mutex device_mutex_;
        mutable std::mutex memory_mutex_;
        
        // Configuration
        bool initialized_;
        int optimization_level_;
        bool profiling_enabled_;
        std::string performance_log_;
    };

    // Factory function for Intel backend
    std::unique_ptr<ComputeBackend> create_intel_backend();

} // namespace ai_vmm