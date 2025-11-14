#pragma once

// Main include file for AI VMM
#include "types.hpp"
#include "backends/compute_backend.hpp"
#include "vmm.hpp"

/**
 * @file ai_vmm.hpp
 * @brief Main header for AI Virtual Machine Manager
 * 
 * This header provides the complete public API for the AI VMM library.
 * Include this file to access all functionality for deploying AI models
 * across heterogeneous hardware accelerators.
 * 
 * @example
 * ```cpp
 * #include <ai_vmm/ai_vmm.hpp>
 * 
 * int main() {
 *     ai_vmm::VMM vmm;
 *     
 *     auto model = vmm.deploy("path/to/model.onnx");
 *     auto input = ai_vmm::Tensor({1, 224, 224, 3});
 *     auto output = model->execute(input);
 *     
 *     return 0;
 * }
 * ```
 */

namespace ai_vmm {
    
/**
 * @brief Library version information
 */
constexpr const char* VERSION = "0.1.0";
constexpr int VERSION_MAJOR = 0;
constexpr int VERSION_MINOR = 1;
constexpr int VERSION_PATCH = 0;

} // namespace ai_vmm