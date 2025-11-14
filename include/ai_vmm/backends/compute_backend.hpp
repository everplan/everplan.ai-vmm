#pragma once

#include "ai_vmm/precision.hpp"
#include <string>
#include <vector>
#include <memory>
#include <cstdint>
#include <functional>
#include <unordered_map>

namespace ai_vmm {

    // Forward declarations
    struct Tensor;
    struct ModelMetadata;
    struct MemoryAllocation;

    // Device types supported across vendors
    enum class DeviceType {
        CPU = 0,
        INTEL_IGPU = 1,
        INTEL_ARC = 2,
        INTEL_NPU = 3,
        NVIDIA_GPU = 4,
        AMD_GPU = 5
    };

    // Memory types for different allocation patterns
    enum class MemoryType {
        HOST,           // Host CPU memory
        DEVICE,         // Device-specific memory (GPU/NPU)
        UNIFIED,        // Unified/shared memory accessible by both
        PINNED          // Pinned host memory for fast transfers
    };

    // Note: Precision enum is defined in types.hpp

    // Device capability information
    struct DeviceInfo {
        std::string name;
        DeviceType type;
        size_t memory_capacity;       // Bytes
        size_t memory_bandwidth;      // Bytes/sec
        std::vector<Precision> supported_precisions;
        double compute_score;         // Relative performance metric
        bool supports_unified_memory;
        std::string driver_version;
    };

    // Memory allocation handle
    struct MemoryAllocation {
        void* ptr;
        size_t size;
        MemoryType type;
        DeviceInfo device;
        
        MemoryAllocation() : ptr(nullptr), size(0) {}
        MemoryAllocation(void* p, size_t s, MemoryType t, const DeviceInfo& d) 
            : ptr(p), size(s), type(t), device(d) {}
    };

    // Tensor representation
    struct Tensor {
        std::vector<int64_t> shape;
        Precision precision;
        MemoryAllocation memory;
        size_t stride_bytes;
        
        // Utility methods
        size_t total_elements() const;
        size_t size_bytes() const;
        bool is_valid() const { return memory.ptr != nullptr; }
    };

    // Model metadata for deployment decisions
    struct ModelMetadata {
        std::string name;
        std::string format;           // "onnx", "gguf", "safetensors", etc.
        size_t parameter_count;
        size_t model_size_bytes;
        std::vector<Precision> required_precisions;
        size_t min_memory_required;
        std::vector<std::string> required_ops;  // Operation types needed
    };

    // Execution context for model inference
    struct ExecutionContext {
        DeviceInfo device;
        std::vector<Tensor> input_tensors;
        std::vector<Tensor> output_tensors;
        std::string model_path;
        Precision execution_precision;
    };

    // Abstract ComputeBackend interface
    class ComputeBackend {
    public:
        virtual ~ComputeBackend() = default;

        // Backend identification
        virtual std::string get_backend_name() const = 0;
        virtual std::string get_version() const = 0;

        // Device management
        virtual bool initialize() = 0;
        virtual void shutdown() = 0;
        virtual std::vector<DeviceInfo> enumerate_devices() = 0;
        virtual bool is_device_available(const DeviceInfo& device) = 0;

        // Memory management
        virtual MemoryAllocation allocate_memory(size_t size, MemoryType type, const DeviceInfo& device) = 0;
        virtual void deallocate_memory(const MemoryAllocation& allocation) = 0;
        virtual bool copy_memory(const MemoryAllocation& src, const MemoryAllocation& dst, size_t size) = 0;
        virtual void synchronize_device(const DeviceInfo& device) = 0;

        // Tensor operations
        virtual Tensor create_tensor(const std::vector<int64_t>& shape, Precision precision, const DeviceInfo& device) = 0;
        virtual void destroy_tensor(const Tensor& tensor) = 0;
        virtual bool copy_tensor(const Tensor& src, Tensor& dst) = 0;

        // Model operations
        virtual bool load_model(const std::string& model_path, const DeviceInfo& device, ExecutionContext& context) = 0;
        virtual bool execute_inference(ExecutionContext& context) = 0;
        virtual void unload_model(ExecutionContext& context) = 0;

        // Performance and capability queries
        virtual bool supports_precision(const DeviceInfo& device, Precision precision) = 0;
        virtual double estimate_inference_time(const ModelMetadata& model, const DeviceInfo& device) = 0;
        virtual size_t get_required_memory(const ModelMetadata& model, const DeviceInfo& device) = 0;

        // Optimization hints
        virtual void set_optimization_level(int level) = 0;  // 0=none, 1=basic, 2=aggressive
        virtual void enable_profiling(bool enable) = 0;
        virtual std::string get_performance_report() = 0;

    protected:
        // Common utility methods for derived classes
        virtual void log_debug(const std::string& message) = 0;
        virtual void log_error(const std::string& message) = 0;
    };

    // Factory function type for backend creation
    using BackendFactory = std::function<std::unique_ptr<ComputeBackend>()>;

    // Backend registry for dynamic loading
    class BackendRegistry {
    public:
        static BackendRegistry& instance();
        
        void register_backend(const std::string& name, BackendFactory factory);
        std::unique_ptr<ComputeBackend> create_backend(const std::string& name);
        std::vector<std::string> list_available_backends();
        
    private:
        std::unordered_map<std::string, BackendFactory> backends_;
    };

} // namespace ai_vmm