#include <gtest/gtest.h>
#include <ai_vmm/vmm.hpp>

class TensorTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

TEST_F(TensorTest, BasicConstruction) {
    ai_vmm::Tensor tensor({1, 224, 224, 3}, ai_vmm::Precision::FP32);
    
    EXPECT_EQ(tensor.shape().size(), 4);
    EXPECT_EQ(tensor.shape()[0], 1);
    EXPECT_EQ(tensor.shape()[1], 224);
    EXPECT_EQ(tensor.shape()[2], 224);
    EXPECT_EQ(tensor.shape()[3], 3);
    EXPECT_EQ(tensor.precision(), ai_vmm::Precision::FP32);
}

TEST_F(TensorTest, SizeCalculation) {
    ai_vmm::Tensor tensor({2, 3, 4}, ai_vmm::Precision::FP32);
    
    EXPECT_EQ(tensor.size(), 24);  // 2 * 3 * 4
    EXPECT_EQ(tensor.byte_size(), 96);  // 24 * 4 bytes (FP32)
}

TEST_F(TensorTest, DifferentPrecisions) {
    ai_vmm::Tensor fp32_tensor({10, 10}, ai_vmm::Precision::FP32);
    ai_vmm::Tensor fp16_tensor({10, 10}, ai_vmm::Precision::FP16);
    ai_vmm::Tensor int8_tensor({10, 10}, ai_vmm::Precision::INT8);
    
    EXPECT_EQ(fp32_tensor.byte_size(), 400);  // 100 * 4
    EXPECT_EQ(fp16_tensor.byte_size(), 200);  // 100 * 2
    EXPECT_EQ(int8_tensor.byte_size(), 100);  // 100 * 1
}

TEST_F(TensorTest, DataPointer) {
    ai_vmm::Tensor tensor({5, 5}, ai_vmm::Precision::FP32);
    
    EXPECT_NE(tensor.data(), nullptr);
    EXPECT_NE(static_cast<const ai_vmm::Tensor&>(tensor).data(), nullptr);
}

TEST_F(TensorTest, EmptyTensor) {
    ai_vmm::Tensor tensor;
    
    EXPECT_EQ(tensor.size(), 1);  // Empty shape defaults to scalar
    EXPECT_EQ(tensor.precision(), ai_vmm::Precision::FP32);
}