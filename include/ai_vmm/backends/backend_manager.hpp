#pragma once

#include "ai_vmm/backends/compute_backend.hpp"
#include "ai_vmm/backends/intel_backend.hpp"
#include "ai_vmm/backends/nvidia_backend.hpp"
#include <memory>
#include <vector>
#include <unordered_map>
#include <mutex>

namespace ai_vmm {

    // Backend manager for handling multiple compute backends
    class BackendManager {
    public:
        BackendManager();
        ~BackendManager();

        // Initialization and lifecycle
        bool initialize();
        void shutdown();
        bool is_initialized() const { return initialized_; }

        // Backend management
        bool register_backend(const std::string& name, std::unique_ptr<ComputeBackend> backend);
        ComputeBackend* get_backend(const std::string& name);
        std::vector<std::string> get_available_backends() const;

        // Device discovery across all backends
        std::vector<DeviceInfo> discover_all_devices();
        std::vector<DeviceInfo> get_devices_by_type(DeviceType type);
        DeviceInfo get_best_device_for_model(const ModelMetadata& model);

        // Cross-backend operations
        bool can_execute_model(const ModelMetadata& model, const DeviceInfo& device);
        double estimate_total_inference_time(const ModelMetadata& model);
        std::vector<DeviceInfo> get_recommended_devices(const ModelMetadata& model, size_t max_devices = 3);

        // Load balancing and resource management
        bool is_device_busy(const DeviceInfo& device);
        void mark_device_busy(const DeviceInfo& device, bool busy);
        double get_device_utilization(const DeviceInfo& device);

        // Configuration
        void set_optimization_level(int level);
        void enable_profiling(bool enable);
        std::string get_performance_summary();

    private:
        mutable std::mutex backends_mutex_;
        std::unordered_map<std::string, std::unique_ptr<ComputeBackend>> backends_;
        std::unordered_map<std::string, bool> device_busy_state_;
        std::unordered_map<std::string, double> device_utilization_;
        
        bool initialized_;
        int global_optimization_level_;
        bool global_profiling_enabled_;

        // Auto-discovery of available backends
        void auto_discover_backends();
        
        // Device scoring for model placement
        double score_device_for_model(const ModelMetadata& model, const DeviceInfo& device);
        
        // Utility functions
        std::string device_to_key(const DeviceInfo& device) const;
        ComputeBackend* find_backend_for_device(const DeviceInfo& device);
    };

} // namespace ai_vmm