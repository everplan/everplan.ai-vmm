#include <gtest/gtest.h>
#include <ai_vmm/types.hpp>

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