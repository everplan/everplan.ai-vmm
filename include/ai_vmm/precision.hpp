#pragma once

namespace ai_vmm {

/**
 * @brief Precision types supported across hardware
 */
enum class Precision {
    FP32,    ///< 32-bit floating point
    FP16,    ///< 16-bit floating point
    BF16,    ///< Brain floating point
    INT8,    ///< 8-bit integer
    INT4,    ///< 4-bit integer
    FP8      ///< 8-bit floating point (future)
};

} // namespace ai_vmm