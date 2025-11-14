#pragma once

#include "types.hpp"
#include "backends/compute_backend.hpp"
#include "backends/backend_manager.hpp"
#include <memory>
#include <vector>
#include <string>

namespace ai_vmm {

// Forward declarations
class Model;
class VMM;

// Note: Tensor class is now defined in backends/compute_backend.hpp

/**
 * @brief Deployed model handle
 */
class DeployedModel {
public:
    DeployedModel(std::shared_ptr<Model> model, std::shared_ptr<ComputeBackend> backend);
    
    /**
     * @brief Execute inference
     * @param input Input tensor
     * @return Output tensor
     */
    Tensor execute(const Tensor& input);
    
    /**
     * @brief Execute inference asynchronously
     * @param input Input tensor
     * @return Future containing output tensor
     */
    std::future<Tensor> execute_async(const Tensor& input);
    
    /**
     * @brief Get execution statistics from last run
     * @return Execution statistics
     */
    ExecutionStats get_last_stats() const { return last_stats_; }
    
    /**
     * @brief Get hardware information for this deployment
     * @return Hardware target information
     */
    HardwareTarget get_hardware_target() const;
    
private:
    std::shared_ptr<Model> model_;
    std::shared_ptr<ComputeBackend> backend_;
    ExecutionStats last_stats_;
};

/**
 * @brief Main VMM class - primary API entry point
 */
class VMM {
public:
    /**
     * @brief Create VMM instance with automatic hardware discovery
     */
    VMM();
    
    /**
     * @brief Create VMM instance with specific backends
     * @param enabled_backends List of hardware types to enable
     */
    explicit VMM(const std::vector<HardwareType>& enabled_backends);
    
    /**
     * @brief Destructor
     */
    ~VMM();
    
    /**
     * @brief Deploy a model from file
     * @param model_path Path to model file (ONNX, OpenVINO IR, etc.)
     * @param constraints Deployment constraints
     * @return Deployed model handle
     */
    std::unique_ptr<DeployedModel> deploy(
        const std::string& model_path,
        const DeploymentConstraints& constraints = {}
    );
    
    /**
     * @brief Deploy a model from Hugging Face Hub
     * @param model_name Model name (e.g., "microsoft/DialoGPT-large")
     * @param constraints Deployment constraints
     * @return Deployed model handle
     */
    std::unique_ptr<DeployedModel> deploy_from_hub(
        const std::string& model_name,
        const DeploymentConstraints& constraints = {}
    );
    
    /**
     * @brief Get list of available hardware
     * @return Vector of available hardware targets
     */
    std::vector<HardwareTarget> get_available_hardware() const;
    
    /**
     * @brief Get recommended hardware for a model
     * @param model_path Path to model file
     * @param constraints Deployment constraints
     * @return Recommended hardware target
     */
    HardwareTarget get_recommended_hardware(
        const std::string& model_path,
        const DeploymentConstraints& constraints = {}
    ) const;
    
    /**
     * @brief Enable debugging and verbose logging
     * @param enable True to enable debug mode
     */
    void set_debug_mode(bool enable);
    
    /**
     * @brief Get VMM version information
     * @return Version string
     */
    static std::string get_version();
    
private:
    class Impl;
    std::unique_ptr<Impl> pimpl_;
};

} // namespace ai_vmm