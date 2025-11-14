#include <ai_vmm/compute_backend.hpp>
#include <memory>
#include <vector>
#include <unordered_map>

namespace ai_vmm {

/**
 * @brief Registry for managing available compute backends
 */
class BackendRegistry {
public:
    static BackendRegistry& instance() {
        static BackendRegistry registry;
        return registry;
    }
    
    void register_backend(std::shared_ptr<ComputeBackend> backend) {
        backends_[backend->get_type()].push_back(backend);
    }
    
    std::vector<std::shared_ptr<ComputeBackend>> get_backends(HardwareType type) const {
        auto it = backends_.find(type);
        return (it != backends_.end()) ? it->second : std::vector<std::shared_ptr<ComputeBackend>>{};
    }
    
    std::vector<std::shared_ptr<ComputeBackend>> get_all_backends() const {
        std::vector<std::shared_ptr<ComputeBackend>> all_backends;
        for (const auto& [type, backend_list] : backends_) {
            all_backends.insert(all_backends.end(), backend_list.begin(), backend_list.end());
        }
        return all_backends;
    }
    
private:
    BackendRegistry() = default;
    std::unordered_map<HardwareType, std::vector<std::shared_ptr<ComputeBackend>>> backends_;
};

} // namespace ai_vmm