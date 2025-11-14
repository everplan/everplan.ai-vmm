#pragma once

#include "types.hpp"
#include <memory>
#include <future>

namespace ai_vmm {

/**
 * @brief Abstract base class for hardware-specific compute backends
 */
class ComputeBackend {
public:
    virtual ~ComputeBackend() = default;
    
    /**
     * @brief Initialize the backend
     * @return True if initialization succeeded
     */
    virtual bool initialize() = 0;
    
    /**
     * @brief Compile a model for this backend
     * @param model_graph The model to compile
     * @return Compiled model handle
     */
    virtual std::shared_ptr<Model> compile(const ModelGraph& model_graph) = 0;
    
    /**
     * @brief Execute a compiled model
     * @param model Compiled model
     * @param input Input tensor
     * @return Future containing the result tensor
     */
    virtual std::future<Tensor> execute(std::shared_ptr<Model> model, const Tensor& input) = 0;
    
    /**
     * @brief Query memory information
     * @return Memory information for this backend
     */
    virtual MemoryInfo query_memory() = 0;
    
    /**
     * @brief Get hardware capabilities
     * @return Capabilities structure
     */
    virtual HardwareCapabilities get_capabilities() = 0;
    
    /**
     * @brief Get backend type
     * @return Hardware type this backend supports
     */
    virtual HardwareType get_type() const = 0;
    
    /**
     * @brief Get device name/identifier
     * @return Human-readable device name
     */
    virtual std::string get_device_name() const = 0;
    
    /**
     * @brief Check if the backend is available and functional
     * @return True if backend can be used
     */
    virtual bool is_available() const = 0;
    
    /**
     * @brief Get performance score for a given model category
     * @param category Model category to evaluate
     * @return Performance score (0.0 - 1.0, higher is better)
     */
    virtual double get_performance_score(ModelCategory category) const = 0;
    
protected:
    /**
     * @brief Validate that the model is compatible with this backend
     * @param model_graph Model to validate
     * @return True if compatible
     */
    virtual bool validate_model_compatibility(const ModelGraph& model_graph) = 0;
    
    /**
     * @brief Apply backend-specific optimizations
     * @param model_graph Model to optimize
     * @return Optimized model graph
     */
    virtual ModelGraph optimize_for_backend(const ModelGraph& model_graph) = 0;
};

} // namespace ai_vmm