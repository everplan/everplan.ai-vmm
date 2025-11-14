#include "ai_vmm/backends/compute_backend.hpp"
#include <unordered_map>
#include <iostream>
#include <numeric>

namespace ai_vmm {

    // Tensor utility implementations
    size_t Tensor::total_elements() const {
        if (shape.empty()) return 0;
        return std::accumulate(shape.begin(), shape.end(), 1ULL, std::multiplies<int64_t>());
    }

    size_t Tensor::size_bytes() const {
        size_t element_count = total_elements();
        if (element_count == 0) return 0;

        size_t bytes_per_element = 0;
        switch (precision) {
            case Precision::FP32: bytes_per_element = 4; break;
            case Precision::FP16: bytes_per_element = 2; break;
            case Precision::BF16: bytes_per_element = 2; break;
            case Precision::INT8: bytes_per_element = 1; break;
            case Precision::INT4: bytes_per_element = 1; break; // Packed, actual is 0.5
            case Precision::FP8: bytes_per_element = 1; break;
        }
        
        return element_count * bytes_per_element;
    }

    // BackendRegistry implementation
    BackendRegistry& BackendRegistry::instance() {
        static BackendRegistry registry;
        return registry;
    }

    void BackendRegistry::register_backend(const std::string& name, BackendFactory factory) {
        backends_[name] = factory;
        std::cout << "[BackendRegistry] Registered backend: " << name << std::endl;
    }

    std::unique_ptr<ComputeBackend> BackendRegistry::create_backend(const std::string& name) {
        auto it = backends_.find(name);
        if (it != backends_.end()) {
            return it->second();
        }
        std::cerr << "[BackendRegistry] Backend not found: " << name << std::endl;
        return nullptr;
    }

    std::vector<std::string> BackendRegistry::list_available_backends() {
        std::vector<std::string> names;
        for (const auto& pair : backends_) {
            names.push_back(pair.first);
        }
        return names;
    }

} // namespace ai_vmm