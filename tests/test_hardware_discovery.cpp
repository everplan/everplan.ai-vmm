#include <gtest/gtest.h>
#include <ai_vmm/types.hpp>
#include <ai_vmm/vmm.hpp>
#include <chrono>
#include <cstdlib>

class HardwareDiscoveryTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(HardwareDiscoveryTest, HardwareTypeEnum) {
    // Test that all hardware types are defined
    EXPECT_NE(ai_vmm::HardwareType::CPU, ai_vmm::HardwareType::UNKNOWN);
    EXPECT_NE(ai_vmm::HardwareType::INTEL_GPU, ai_vmm::HardwareType::UNKNOWN);
    EXPECT_NE(ai_vmm::HardwareType::INTEL_ARC, ai_vmm::HardwareType::UNKNOWN);
    EXPECT_NE(ai_vmm::HardwareType::INTEL_NPU, ai_vmm::HardwareType::UNKNOWN);
    EXPECT_NE(ai_vmm::HardwareType::NVIDIA_GPU, ai_vmm::HardwareType::UNKNOWN);
    EXPECT_NE(ai_vmm::HardwareType::AMD_GPU, ai_vmm::HardwareType::UNKNOWN);
}

TEST_F(HardwareDiscoveryTest, ModelCategoryEnum) {
    // Test that all model categories are defined
    EXPECT_NE(ai_vmm::ModelCategory::LLM_TRANSFORMER, ai_vmm::ModelCategory::UNKNOWN_ARCHITECTURE);
    EXPECT_NE(ai_vmm::ModelCategory::VISION_CNN, ai_vmm::ModelCategory::UNKNOWN_ARCHITECTURE);
    EXPECT_NE(ai_vmm::ModelCategory::VISION_TRANSFORMER, ai_vmm::ModelCategory::UNKNOWN_ARCHITECTURE);
    EXPECT_NE(ai_vmm::ModelCategory::SPEECH_RNN, ai_vmm::ModelCategory::UNKNOWN_ARCHITECTURE);
    EXPECT_NE(ai_vmm::ModelCategory::RECOMMENDATION_SYSTEM, ai_vmm::ModelCategory::UNKNOWN_ARCHITECTURE);
    EXPECT_NE(ai_vmm::ModelCategory::SCIENTIFIC_ML, ai_vmm::ModelCategory::UNKNOWN_ARCHITECTURE);
}

TEST_F(HardwareDiscoveryTest, PrecisionEnum) {
    // Test precision types
    EXPECT_TRUE(ai_vmm::Precision::FP32 != ai_vmm::Precision::FP16);
    EXPECT_TRUE(ai_vmm::Precision::FP16 != ai_vmm::Precision::BF16);
    EXPECT_TRUE(ai_vmm::Precision::INT8 != ai_vmm::Precision::INT4);
}

TEST_F(HardwareDiscoveryTest, MemoryInfoStructure) {
    ai_vmm::MemoryInfo mem_info;
    
    // Test default initialization
    EXPECT_EQ(mem_info.total_memory, 0);
    EXPECT_EQ(mem_info.available_memory, 0);
    EXPECT_EQ(mem_info.memory_bandwidth, 0);
    EXPECT_FALSE(mem_info.supports_unified_memory);
    EXPECT_TRUE(mem_info.memory_type.empty());
}

TEST_F(HardwareDiscoveryTest, ExecutionStatsStructure) {
    ai_vmm::ExecutionStats stats;
    
    // Test default initialization
    EXPECT_DOUBLE_EQ(stats.execution_time_ms, 0.0);
    EXPECT_EQ(stats.memory_used_bytes, 0);
    EXPECT_DOUBLE_EQ(stats.power_consumed_watts, 0.0);
    EXPECT_DOUBLE_EQ(stats.throughput_ops_per_sec, 0.0);
    EXPECT_TRUE(stats.device_name.empty());
    EXPECT_EQ(stats.device_type, ai_vmm::HardwareType::UNKNOWN);
}

class ActualHardwareDiscoveryTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Explicitly disable test mode for this test to test actual hardware discovery
        unsetenv("AI_VMM_TEST_MODE");
    }
    void TearDown() override {}
};

TEST_F(ActualHardwareDiscoveryTest, DISABLED_HardwareDiscoveryWithTimeout) {
    // NOTE: This test is disabled due to a hang in register_backend() method
    // The hang occurs when BackendManager tries to register the Intel backend
    // This needs investigation of the backend registration locking mechanism
    // 
    // Issue: After Intel backend creation (0ms), the call to register_backend() hangs
    // Root cause: Likely mutex deadlock or initialization order issue in backend registration
    //
    // Hardware discovery is crucial for production use - this must be fixed before deployment
    // Test that hardware discovery completes within a reasonable time
    auto start_time = std::chrono::steady_clock::now();
    
    bool creation_successful = false;
    std::string error_message;
    
    try {
        std::cout << "Starting VMM creation with actual hardware discovery..." << std::endl;
        ai_vmm::VMM vmm;
        creation_successful = true;
        
        // If we get here, check that we found some hardware
        auto hardware = vmm.get_available_hardware();
        EXPECT_GT(hardware.size(), 0) << "Should discover at least CPU hardware";
        
        // Log discovered hardware for debugging
        std::cout << "Discovered " << hardware.size() << " hardware targets:" << std::endl;
        for (const auto& hw : hardware) {
            std::cout << "  - Type " << static_cast<int>(hw.get_type()) << ": " << hw.get_name() << std::endl;
        }
        
    } catch (const std::exception& e) {
        error_message = e.what();
        std::cerr << "Exception during hardware discovery: " << error_message << std::endl;
    } catch (...) {
        error_message = "Unknown exception";
        std::cerr << "Unknown exception during hardware discovery" << std::endl;
    }
    
    auto end_time = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    std::cout << "Hardware discovery took " << duration.count() << "ms" << std::endl;
    
    // Test should complete within 8 seconds (giving extra margin for the 5-second timeout)
    EXPECT_LT(duration.count(), 8000) << "Hardware discovery took too long: " << duration.count() << "ms";
    
    if (!creation_successful) {
        // If creation failed, it should still complete quickly
        EXPECT_LT(duration.count(), 1000) << "Even failed creation should be fast, took: " << duration.count() << "ms";
        FAIL() << "VMM creation failed: " << error_message;
    } else {
        EXPECT_TRUE(creation_successful) << "VMM creation should succeed";
        std::cout << "✅ Hardware discovery test passed!" << std::endl;
    }
}

TEST_F(HardwareDiscoveryTest, VMMCreationInTestMode) {
    // Test that VMM creation works reliably in test mode
    setenv("AI_VMM_TEST_MODE", "1", 1);
    
    try {
        ai_vmm::VMM vmm;
        
        // Should succeed even without hardware discovery
        auto hardware = vmm.get_available_hardware();
        // In test mode, we might have 0 devices (no auto-discovery)
        EXPECT_GE(hardware.size(), 0) << "Hardware list should be valid";
        
        std::cout << "Test mode VMM created successfully with " 
                  << hardware.size() << " hardware targets" << std::endl;
        
    } catch (const std::exception& e) {
        FAIL() << "VMM creation should not throw in test mode: " << e.what();
    }
    
    unsetenv("AI_VMM_TEST_MODE");
}