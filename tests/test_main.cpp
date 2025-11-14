#include <gtest/gtest.h>

// Main test entry point - Google Test will handle the rest
int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}