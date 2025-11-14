#pragma once

#include "precision.hpp"
#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <future>

namespace ai_vmm {

// Forward declarations
class Model;
class Tensor;
class ComputeBackend;
class HardwareTarget;
class DeploymentConstraints;
class ModelGraph;
class ExecutionPlan;

/**
 * @brief Hardware type enumeration
 */
enum class HardwareType {
    CPU,           ///< Intel/AMD CPUs
    INTEL_GPU,     ///< Intel Integrated Graphics (legacy name)
    INTEL_IGPU,    ///< Intel Integrated Graphics
    INTEL_ARC,     ///< Intel ARC Discrete GPU
    INTEL_NPU,     ///< Intel NPU (AI Accelerator)
    NVIDIA_GPU,    ///< NVIDIA CUDA devices
    AMD_GPU,       ///< AMD ROCm devices
    QUALCOMM_NPU,  ///< Qualcomm NPU
    APPLE_SILICON, ///< Apple Neural Engine
    UNKNOWN        ///< Unknown or unsupported hardware
};

/**
 * @brief Model category for optimization purposes
 */
enum class ModelCategory {
    LLM_TRANSFORMER,      ///< Large Language Models, GPT-style
    VISION_CNN,           ///< Convolutional Neural Networks
    VISION_TRANSFORMER,   ///< Vision Transformers (ViT)
    SPEECH_RNN,           ///< RNN/LSTM for speech processing
    RECOMMENDATION_SYSTEM,///< Embedding-heavy recommender models
    SCIENTIFIC_ML,        ///< Physics-informed, GNNs, etc.
    UNKNOWN_ARCHITECTURE  ///< Unknown model type
};

// Note: Precision enum is now defined in backends/compute_backend.hpp

/**
 * @brief Hardware capabilities structure
 */
struct HardwareCapabilities {
    // Compute characteristics
    bool fast_attention_ops = false;      ///< Good for transformers
    bool fast_conv_ops = false;           ///< Good for CNNs
    bool fast_rnn_ops = false;            ///< Good for sequential models
    bool large_embedding_support = false; ///< Good for recommendation
    bool tensor_cores = false;            ///< NVIDIA Tensor Cores or equivalent
    
    // Memory characteristics
    size_t memory_bandwidth = 0;          ///< Memory bandwidth in GB/s
    size_t memory_capacity = 0;           ///< Memory capacity in bytes
    bool unified_memory = false;          ///< Supports unified memory
    
    // Precision support
    std::vector<Precision> supported_precisions;
    
    // Vendor-specific features
    std::unordered_map<std::string, std::string> vendor_extensions;
};

/**
 * @brief Hardware target information
 */
class HardwareTarget {
public:
    HardwareTarget(HardwareType type, const std::string& name)
        : type_(type), name_(name) {}
    
    HardwareType get_type() const { return type_; }
    const std::string& get_name() const { return name_; }
    
    void set_capabilities(const HardwareCapabilities& caps) { capabilities_ = caps; }
    const HardwareCapabilities& get_capabilities() const { return capabilities_; }
    
    bool operator==(const HardwareTarget& other) const {
        return type_ == other.type_ && name_ == other.name_;
    }
    
private:
    HardwareType type_;
    std::string name_;
    HardwareCapabilities capabilities_;
};

/**
 * @brief Deployment constraints for model optimization
 */
struct DeploymentConstraints {
    // Performance constraints
    uint32_t max_latency_ms = 0;          ///< Maximum inference latency
    uint32_t min_throughput = 0;          ///< Minimum throughput (inferences/sec)
    
    // Resource constraints
    uint32_t max_memory_mb = 0;           ///< Maximum memory usage
    uint32_t power_budget_watts = 0;      ///< Power budget in watts
    
    // Hardware preferences
    std::vector<HardwareType> preferred_hardware;
    std::vector<HardwareType> excluded_hardware;
    
    // Quality constraints
    Precision min_precision = Precision::FP32;
    float max_accuracy_loss = 0.05f;      ///< Maximum acceptable accuracy loss
    
    // Deployment mode
    bool batch_mode = false;              ///< Optimize for batch vs single inference
    uint32_t max_batch_size = 1;
};

/**
 * @brief Memory information for a hardware target
 */
struct MemoryInfo {
    size_t total_memory = 0;
    size_t available_memory = 0;
    size_t memory_bandwidth = 0;
    bool supports_unified_memory = false;
    std::string memory_type; // "DDR4", "GDDR6", "HBM2", etc.
};

/**
 * @brief Execution statistics
 */
struct ExecutionStats {
    double execution_time_ms = 0.0;
    size_t memory_used_bytes = 0;
    double power_consumed_watts = 0.0;
    double throughput_ops_per_sec = 0.0;
    std::string device_name;
    HardwareType device_type = HardwareType::UNKNOWN;
};

} // namespace ai_vmm