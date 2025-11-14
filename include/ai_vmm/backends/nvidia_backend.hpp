#pragma once

#include "ai_vmm/backends/compute_backend.hpp"
#include <memory>
#include <mutex>

namespace ai_vmm {

    // NVIDIA-specific device information
    struct NVIDIADeviceInfo {
        DeviceInfo base;
        std::string device_id;
        int cuda_major_version;
        int cuda_minor_version;
        std::string cuda_driver_version;
        std::string cudnn_version;
        
        // GPU-specific information
        int sm_count;                  // Streaming Multiprocessor count
        int compute_capability_major;
        int compute_capability_minor;
        size_t shared_memory_per_sm;
        size_t max_threads_per_block;
        std::string architecture;      // "Ada Lovelace", "Ampere", etc.
        
        // Performance characteristics
        double tensor_performance_fp16;  // TFLOPS at FP16
        double tensor_performance_int8;  // TOPS at INT8
        bool supports_tensor_cores;
        bool supports_mig;               // Multi-Instance GPU
    };

    // NVIDIA backend implementation
    class NVIDIABackend : public ComputeBackend {
    public:
        NVIDIABackend();
        virtual ~NVIDIABackend();

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
        std::vector<NVIDIADeviceInfo> discover_nvidia_devices();
        NVIDIADeviceInfo create_nvidia_device_info(int device_id);
        
        // CUDA capability detection
        bool check_cuda_availability();
        bool check_cudnn_availability();
        std::string get_cuda_version();
        std::string get_cudnn_version();
        
        // Memory management helpers
        void* allocate_cuda_memory(size_t size, int device_id);
        void* allocate_host_pinned_memory(size_t size);
        void* allocate_unified_memory(size_t size, int device_id);
        
        // Model loading helpers
        bool load_tensorrt_model(const std::string& model_path, const DeviceInfo& device, ExecutionContext& context);
        bool load_onnx_with_tensorrt(const std::string& model_path, const DeviceInfo& device, ExecutionContext& context);
        
        // Performance estimation
        double estimate_gpu_inference_time(const ModelMetadata& model, const NVIDIADeviceInfo& device);
        double get_theoretical_throughput(const NVIDIADeviceInfo& device, Precision precision);
        
        // CUDA stream management
        void* create_cuda_stream(int device_id);
        void destroy_cuda_stream(void* stream);
        
        // Thread safety
        mutable std::mutex device_mutex_;
        mutable std::mutex memory_mutex_;
        
        // Configuration
        bool initialized_;
        int optimization_level_;
        bool profiling_enabled_;
        std::string performance_log_;
        
        // CUDA contexts per device
        std::unordered_map<int, void*> cuda_contexts_;
        std::unordered_map<int, void*> cuda_streams_;
    };

    // Factory function for NVIDIA backend
    std::unique_ptr<ComputeBackend> create_nvidia_backend();

} // namespace ai_vmm