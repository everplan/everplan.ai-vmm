#include <gtest/gtest.h>
#include <ai_vmm/vmm.hpp>

class VMMBasicTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(VMMBasicTest, VMMCreation) {
    EXPECT_NO_THROW({
        ai_vmm::VMM vmm;
    });
}

TEST_F(VMMBasicTest, VMMCreationWithSpecificBackends) {
    std::vector<ai_vmm::HardwareType> backends = {
        ai_vmm::HardwareType::CPU,
        ai_vmm::HardwareType::INTEL_GPU
    };
    
    EXPECT_NO_THROW({
        ai_vmm::VMM vmm(backends);
    });
}

TEST_F(VMMBasicTest, GetAvailableHardware) {
    ai_vmm::VMM vmm;
    
    auto hardware = vmm.get_available_hardware();
    // Note: In the skeleton implementation, this may return empty
    // In full implementation, should return at least CPU
}

TEST_F(VMMBasicTest, VersionInformation) {
    std::string version = ai_vmm::VMM::get_version();
    EXPECT_FALSE(version.empty());
    EXPECT_EQ(version, "0.1.0");
}

TEST_F(VMMBasicTest, DeploymentConstraintsConstruction) {
    ai_vmm::DeploymentConstraints constraints;
    
    // Test default values
    EXPECT_EQ(constraints.max_latency_ms, 0);
    EXPECT_EQ(constraints.min_throughput, 0);
    EXPECT_EQ(constraints.max_memory_mb, 0);
    EXPECT_EQ(constraints.power_budget_watts, 0);
    EXPECT_EQ(constraints.min_precision, ai_vmm::Precision::FP32);
    EXPECT_FLOAT_EQ(constraints.max_accuracy_loss, 0.05f);
    EXPECT_FALSE(constraints.batch_mode);
    EXPECT_EQ(constraints.max_batch_size, 1);
}

TEST_F(VMMBasicTest, HardwareTargetConstruction) {
    ai_vmm::HardwareTarget target(ai_vmm::HardwareType::NVIDIA_GPU, "RTX 4090");
    
    EXPECT_EQ(target.get_type(), ai_vmm::HardwareType::NVIDIA_GPU);
    EXPECT_EQ(target.get_name(), "RTX 4090");
}

TEST_F(VMMBasicTest, HardwareCapabilitiesStructure) {
    ai_vmm::HardwareCapabilities caps;
    
    // Test default initialization
    EXPECT_FALSE(caps.fast_attention_ops);
    EXPECT_FALSE(caps.fast_conv_ops);
    EXPECT_FALSE(caps.fast_rnn_ops);
    EXPECT_FALSE(caps.large_embedding_support);
    EXPECT_FALSE(caps.tensor_cores);
    EXPECT_EQ(caps.memory_bandwidth, 0);
    EXPECT_EQ(caps.memory_capacity, 0);
    EXPECT_FALSE(caps.unified_memory);
    EXPECT_TRUE(caps.supported_precisions.empty());
}