#include <ai_vmm/vmm.hpp>
#include <ai_vmm/types.hpp>
#include <ai_vmm/ai_vmm.hpp>
#include <ai_vmm/backends/backend_manager.hpp>

#include <memory>
#include <vector>
#include <unordered_map>
#include <stdexcept>
#include <iostream>
#include <filesystem>
#include <algorithm>
#include <chrono>
#include <future>

// Forward declare HardwareDiscovery
namespace ai_vmm {
    class HardwareDiscovery;
}

namespace ai_vmm {

// Simple stub implementations for PIMPL pattern
class ModelLoader {
public:
    ModelLoader() = default;
    ~ModelLoader() = default;
};

class HardwareRegistry {
public:
    HardwareRegistry() = default;
    ~HardwareRegistry() = default;
};

class ModelAnalyzer {
public:
    ModelAnalyzer() = default;
    ~ModelAnalyzer() = default;
};

class WorkloadScheduler {
public:
    WorkloadScheduler() = default;
    ~WorkloadScheduler() = default;
};

// Concrete DeployedModel implementation
class DeployedModelImpl : public DeployedModel {
public:
    DeployedModelImpl(ComputeBackend* backend, const ExecutionContext& context, const ModelMetadata& metadata)
        : DeployedModel(nullptr, nullptr), backend_(backend), context_(context), metadata_(metadata) {}
    
    ~DeployedModelImpl() {
        if (backend_) {
            backend_->unload_model(context_);
        }
    }
    
    Tensor execute(const Tensor& input) {
        if (!backend_) {
            throw std::runtime_error("Backend not available");
        }
        
        // Create mutable copy of context for execution
        ExecutionContext exec_context = context_;
        exec_context.input_tensors = {input};
        exec_context.output_tensors.clear();
        
        auto start_time = std::chrono::high_resolution_clock::now();
        
        bool success = backend_->execute_inference(exec_context);
        
        auto end_time = std::chrono::high_resolution_clock::now();
        
        if (!success) {
            throw std::runtime_error("Inference execution failed");
        }
        
        if (exec_context.output_tensors.empty()) {
            throw std::runtime_error("No output tensors produced");
        }
        
        // Update execution statistics
        last_stats_.execution_time_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time).count();
        last_stats_.memory_used_bytes = metadata_.model_size_bytes; // Simplified
        last_stats_.device_name = context_.device.name;
        last_stats_.device_type = convert_device_type_to_hardware_type(context_.device.type);
        
        return exec_context.output_tensors[0];
    }
    
    std::future<Tensor> execute_async(const Tensor& input) {
        return std::async(std::launch::async, [this, input]() {
            return execute(input);
        });
    }
    
    HardwareTarget get_hardware_target() const {
        HardwareType hw_type = convert_device_type_to_hardware_type(context_.device.type);
        return HardwareTarget(hw_type, context_.device.name);
    }
    
    ExecutionStats get_last_stats() const {
        return last_stats_;
    }
    
private:
    ComputeBackend* backend_;
    ExecutionContext context_;
    ModelMetadata metadata_;
    mutable ExecutionStats last_stats_;
    
    HardwareType convert_device_type_to_hardware_type(DeviceType device_type) const {
        switch (device_type) {
            case DeviceType::CPU: return HardwareType::CPU;
            case DeviceType::INTEL_IGPU: return HardwareType::INTEL_IGPU;
            case DeviceType::INTEL_ARC: return HardwareType::INTEL_ARC;
            case DeviceType::INTEL_NPU: return HardwareType::INTEL_NPU;
            case DeviceType::NVIDIA_GPU: return HardwareType::NVIDIA_GPU;
            case DeviceType::AMD_GPU: return HardwareType::AMD_GPU;
            default: return HardwareType::UNKNOWN;
        }
    }
};

// Forward declarations for internal classes
class ModelLoader;
class HardwareRegistry;
class ModelAnalyzer;
class WorkloadScheduler;

/**
 * @brief VMM implementation (PIMPL pattern)
 */
class VMM::Impl {
public:
    Impl(const std::vector<HardwareType>& enabled_backends);
    ~Impl();
    
    std::unique_ptr<DeployedModel> deploy(
        const std::string& model_path,
        const DeploymentConstraints& constraints
    );
    
    std::unique_ptr<DeployedModel> deploy_from_hub(
        const std::string& model_name,
        const DeploymentConstraints& constraints
    );
    
    std::vector<HardwareTarget> get_available_hardware() const;
    
    HardwareTarget get_recommended_hardware(
        const std::string& model_path,
        const DeploymentConstraints& constraints
    ) const;
    
    void set_debug_mode(bool enable);
    
private:
    void initialize_backends(const std::vector<HardwareType>& enabled_backends);
    void discover_hardware();
    DeviceInfo select_best_device(
        const ModelMetadata& model_metadata,
        const DeploymentConstraints& constraints
    ) const;
    
    // Model deployment helper methods
    ModelMetadata extract_model_metadata(const std::string& model_path) const;
    Precision select_precision(
        const ModelMetadata& metadata,
        const DeviceInfo& device,
        const DeploymentConstraints& constraints
    ) const;
    ComputeBackend* find_backend_for_device(const DeviceInfo& device) const;
    
    // New backend management system
    std::unique_ptr<BackendManager> backend_manager_;
    
    // Legacy components (simplified stubs)
    std::unique_ptr<ModelLoader> model_loader_;
    std::unique_ptr<HardwareRegistry> hardware_registry_;
    std::unique_ptr<ModelAnalyzer> model_analyzer_;
    std::unique_ptr<WorkloadScheduler> scheduler_;
    
    std::vector<HardwareTarget> available_hardware_;
    bool debug_mode_ = false;
    bool initialized_ = false;
};

VMM::Impl::Impl(const std::vector<HardwareType>& enabled_backends) {
    try {
        // Initialize the new backend management system
        backend_manager_ = std::make_unique<BackendManager>();
        
        // Initialize stub implementations for compatibility
        model_loader_ = std::make_unique<ModelLoader>();
        hardware_registry_ = std::make_unique<HardwareRegistry>();
        model_analyzer_ = std::make_unique<ModelAnalyzer>();
        scheduler_ = std::make_unique<WorkloadScheduler>();
        
        initialize_backends(enabled_backends);
        discover_hardware();
        initialized_ = true;
        
        if (debug_mode_) {
            auto backends = backend_manager_->get_available_backends();
            std::cout << "AI VMM initialized with " << backends.size() 
                      << " backends and " << available_hardware_.size() 
                      << " hardware targets" << std::endl;
        }
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to initialize AI VMM: " + std::string(e.what()));
    }
}

VMM::Impl::~Impl() {
    // Cleanup will be handled by smart pointers
}

void VMM::Impl::initialize_backends(const std::vector<HardwareType>& enabled_backends) {
    if (debug_mode_) {
        std::cout << "Initializing backend manager..." << std::endl;
    }
    
    // Initialize the backend manager which auto-discovers available backends
    if (!backend_manager_->initialize()) {
        throw std::runtime_error("Failed to initialize backend manager");
    }
    
    // Log available backends
    auto available_backends = backend_manager_->get_available_backends();
    if (debug_mode_) {
        std::cout << "Initialized backends: ";
        for (const auto& backend_name : available_backends) {
            std::cout << backend_name << " ";
        }
        std::cout << std::endl;
    }
}

void VMM::Impl::discover_hardware() {
    if (debug_mode_) {
        std::cout << "Discovering available hardware..." << std::endl;
    }
    
    // Get devices from the backend manager
    auto devices = backend_manager_->discover_all_devices();
    
    // Convert new DeviceInfo to legacy HardwareTarget format
    available_hardware_.clear();
    for (const auto& device : devices) {
        HardwareTarget hw_target(HardwareType::UNKNOWN, "Unknown");
        
        // Convert DeviceType to HardwareType
        switch (device.type) {
            case DeviceType::CPU:
                hw_target = HardwareTarget(HardwareType::CPU, device.name);
                break;
            case DeviceType::INTEL_IGPU:
                hw_target = HardwareTarget(HardwareType::INTEL_IGPU, device.name);
                break;
            case DeviceType::INTEL_ARC:
                hw_target = HardwareTarget(HardwareType::INTEL_ARC, device.name);
                break;
            case DeviceType::INTEL_NPU:
                hw_target = HardwareTarget(HardwareType::INTEL_NPU, device.name);
                break;
            case DeviceType::NVIDIA_GPU:
                hw_target = HardwareTarget(HardwareType::NVIDIA_GPU, device.name);
                break;
            case DeviceType::AMD_GPU:
                hw_target = HardwareTarget(HardwareType::AMD_GPU, device.name);
                break;
        }
        
        available_hardware_.push_back(hw_target);
    }
    
    if (debug_mode_) {
        std::cout << "Found " << available_hardware_.size() << " hardware targets:" << std::endl;
        for (const auto& hw : available_hardware_) {
            std::cout << "  - " << hw.get_name() 
                      << " (Type: " << static_cast<int>(hw.get_type()) << ")" << std::endl;
        }
    }
}

std::unique_ptr<DeployedModel> VMM::Impl::deploy(
    const std::string& model_path,
    const DeploymentConstraints& constraints
) {
    if (!initialized_) {
        throw std::runtime_error("VMM not properly initialized");
    }
    
    if (debug_mode_) {
        std::cout << "Deploying model: " << model_path << std::endl;
    }
    
    // Step 1: Validate model file exists
    if (!std::filesystem::exists(model_path)) {
        throw std::runtime_error("Model file not found: " + model_path);
    }
    
    // Step 2: Extract model metadata
    ModelMetadata metadata = extract_model_metadata(model_path);
    if (debug_mode_) {
        std::cout << "Model metadata extracted - Format: " << metadata.format
                  << ", Memory required: " << metadata.model_size_bytes << " bytes" << std::endl;
    }
    
    // Step 3: Select optimal device for deployment  
    DeviceInfo selected_device = backend_manager_->get_best_device_for_model(metadata);
    if (debug_mode_) {
        std::cout << "Selected device: " << selected_device.name 
                  << " (Type: " << static_cast<int>(selected_device.type) << ")" << std::endl;
    }
    
    // Step 4: Find backend that owns this device
    ComputeBackend* backend = find_backend_for_device(selected_device);
    if (!backend) {
        throw std::runtime_error("No backend available for device: " + selected_device.name);
    }
    
    // Step 5: Create execution context and load model
    ExecutionContext context;
    context.device = selected_device;
    context.model_path = model_path;
    context.execution_precision = select_precision(metadata, selected_device, constraints);
    
    if (!backend->load_model(model_path, selected_device, context)) {
        throw std::runtime_error("Failed to load model on device: " + selected_device.name);
    }
    
    // Step 6: Create and return deployed model
    auto deployed_model = std::make_unique<DeployedModelImpl>(backend, context, metadata);
    
    if (debug_mode_) {
        std::cout << "Model deployed successfully on " << selected_device.name << std::endl;
    }
    
    return deployed_model;
}

// Helper method implementations
ModelMetadata VMM::Impl::extract_model_metadata(const std::string& model_path) const {
    ModelMetadata metadata{};
    
    // Extract file extension to determine format
    auto extension_pos = model_path.find_last_of('.');
    if (extension_pos == std::string::npos) {
        throw std::runtime_error("Model file has no extension: " + model_path);
    }
    
    std::string extension = model_path.substr(extension_pos + 1);
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);
    
    // Set basic metadata based on file type
    if (extension == "onnx") {
        metadata.format = "ONNX";
    } else if (extension == "xml" || extension == "bin") {
        metadata.format = "OpenVINO";
    } else {
        throw std::runtime_error("Unsupported model format: " + extension);
    }
    
    // Extract model name from path
    auto filename_start = model_path.find_last_of("/\\");
    if (filename_start != std::string::npos) {
        metadata.name = model_path.substr(filename_start + 1, extension_pos - filename_start - 1);
    } else {
        metadata.name = model_path.substr(0, extension_pos);
    }
    
    // Estimate model size and memory requirements
    std::error_code ec;
    auto file_size = std::filesystem::file_size(model_path, ec);
    if (!ec) {
        metadata.model_size_bytes = file_size;
        // Rough estimate: 3x file size for memory (weights + activations + workspace)
        metadata.min_memory_required = file_size * 3;
    } else {
        metadata.model_size_bytes = 100 * 1024 * 1024; // 100MB default
        metadata.min_memory_required = 300 * 1024 * 1024; // 300MB default
    }
    
    // Set default precision requirements
    metadata.required_precisions = {Precision::FP32};
    
    // Add some common required operations based on format
    if (extension == "onnx") {
        metadata.required_ops = {"MatMul", "Add", "Relu"};
    }
    
    return metadata;
}

Precision VMM::Impl::select_precision(
    const ModelMetadata& metadata,
    const DeviceInfo& device,
    const DeploymentConstraints& constraints
) const {
    // Honor explicit precision requirement if specified
    if (constraints.min_precision != Precision::FP32) { // Use min_precision from constraints
        return constraints.min_precision;
    }
    
    // Select based on device capabilities
    auto device_backend = find_backend_for_device(device);
    if (!device_backend) {
        return Precision::FP32; // Fallback
    }
    
    // Try required precisions to find what the device supports
    for (auto precision : metadata.required_precisions) {
        if (device_backend->supports_precision(device, precision)) {
            return precision;
        }
    }
    
    // If no required precision is supported, try common precisions
    std::vector<Precision> common_precisions = {Precision::FP16, Precision::FP32};
    for (auto precision : common_precisions) {
        if (device_backend->supports_precision(device, precision)) {
            return precision;
        }
    }
    
    return Precision::FP32; // Final fallback
}

ComputeBackend* VMM::Impl::find_backend_for_device(const DeviceInfo& device) const {
    // Try each backend to see which one owns this device
    auto backend_names = backend_manager_->get_available_backends();
    
    for (const auto& name : backend_names) {
        ComputeBackend* backend = backend_manager_->get_backend(name);
        if (backend && backend->is_device_available(device)) {
            return backend;
        }
    }
    
    return nullptr;
}

std::unique_ptr<DeployedModel> VMM::Impl::deploy_from_hub(
    const std::string& model_name,
    const DeploymentConstraints& constraints
) {
    if (debug_mode_) {
        std::cout << "Deploying model from hub: " << model_name << std::endl;
    }
    
    // TODO: Download from Hugging Face Hub and deploy
    throw std::runtime_error("Hub deployment not yet implemented");
}

std::vector<HardwareTarget> VMM::Impl::get_available_hardware() const {
    return available_hardware_;
}

HardwareTarget VMM::Impl::get_recommended_hardware(
    const std::string& model_path,
    const DeploymentConstraints& constraints
) const {
    if (available_hardware_.empty()) {
        throw std::runtime_error("No hardware available");
    }
    
    // TODO: Implement intelligent hardware recommendation
    // For now, return first available hardware
    return available_hardware_[0];
}

void VMM::Impl::set_debug_mode(bool enable) {
    debug_mode_ = enable;
    if (enable) {
        std::cout << "AI VMM debug mode enabled" << std::endl;
    }
}

// VMM public interface implementation

VMM::VMM() : VMM({HardwareType::CPU, HardwareType::INTEL_GPU, HardwareType::INTEL_NPU, HardwareType::NVIDIA_GPU}) {
}

VMM::VMM(const std::vector<HardwareType>& enabled_backends) 
    : pimpl_(std::make_unique<Impl>(enabled_backends)) {
}

VMM::~VMM() = default;

std::unique_ptr<DeployedModel> VMM::deploy(
    const std::string& model_path,
    const DeploymentConstraints& constraints
) {
    return pimpl_->deploy(model_path, constraints);
}

std::unique_ptr<DeployedModel> VMM::deploy_from_hub(
    const std::string& model_name,
    const DeploymentConstraints& constraints
) {
    return pimpl_->deploy_from_hub(model_name, constraints);
}

std::vector<HardwareTarget> VMM::get_available_hardware() const {
    return pimpl_->get_available_hardware();
}

HardwareTarget VMM::get_recommended_hardware(
    const std::string& model_path,
    const DeploymentConstraints& constraints
) const {
    return pimpl_->get_recommended_hardware(model_path, constraints);
}

void VMM::set_debug_mode(bool enable) {
    pimpl_->set_debug_mode(enable);
}

std::string VMM::get_version() {
    return ai_vmm::VERSION;
}

// DeployedModel implementation

DeployedModel::DeployedModel(std::shared_ptr<Model> model, std::shared_ptr<ComputeBackend> backend)
    : model_(model), backend_(backend) {
}

Tensor DeployedModel::execute(const Tensor& input) {
    // TODO: Implement synchronous execution
    throw std::runtime_error("Execution not yet implemented");
}

std::future<Tensor> DeployedModel::execute_async(const Tensor& input) {
    // TODO: Implement asynchronous execution
    return std::async(std::launch::async, [this, input]() {
        return execute(input);
    });
}

HardwareTarget DeployedModel::get_hardware_target() const {
    // TODO: Return actual hardware target from backend
    return HardwareTarget(HardwareType::UNKNOWN, "Unknown");
}

} // namespace ai_vmm