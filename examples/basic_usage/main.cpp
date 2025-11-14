#include <ai_vmm/ai_vmm.hpp>
#include <iostream>
#include <chrono>

int main() {
    try {
        std::cout << "AI VMM Basic Usage Example" << std::endl;
        std::cout << "Version: " << ai_vmm::VMM::get_version() << std::endl;
        
        // Initialize VMM with automatic hardware discovery
        ai_vmm::VMM vmm;
        vmm.set_debug_mode(true);
        
        // Get available hardware
        auto available_hardware = vmm.get_available_hardware();
        std::cout << "\nAvailable hardware:" << std::endl;
        for (const auto& hw : available_hardware) {
            std::cout << "  - " << hw.get_name() 
                      << " (Type: " << static_cast<int>(hw.get_type()) << ")" 
                      << std::endl;
        }
        
        // Example 1: Deploy a simple model with default constraints
        std::cout << "\nExample 1: Basic deployment with MobileNet V2" << std::endl;
        try {
            auto model = vmm.deploy("models/mobilenetv2.onnx");
            std::cout << "Model deployed successfully" << std::endl;
            
            // Create example input tensor
            ai_vmm::Tensor input({1, 224, 224, 3}, ai_vmm::Precision::FP32);
            
            // Execute inference
            auto start = std::chrono::high_resolution_clock::now();
            auto output = model->execute(input);
            auto end = std::chrono::high_resolution_clock::now();
            
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
            std::cout << "Inference completed in " << duration.count() << " ms" << std::endl;
            
        } catch (const std::exception& e) {
            std::cout << "Model deployment failed: " << e.what() << std::endl;
        }
        
        // Example 2: Deploy with constraints
        std::cout << "\nExample 2: Deployment with constraints" << std::endl;
        ai_vmm::DeploymentConstraints constraints;
        constraints.max_latency_ms = 100;
        constraints.power_budget_watts = 150;
        constraints.preferred_hardware = {ai_vmm::HardwareType::NVIDIA_GPU, ai_vmm::HardwareType::INTEL_NPU};
        
        try {
            auto constrained_model = vmm.deploy("models/mobilenetv2.onnx", constraints);
            std::cout << "Constrained model deployed successfully" << std::endl;
            
            // Test inference with constraints
            ai_vmm::Tensor input({1, 224, 224, 3}, ai_vmm::Precision::FP32);
            auto output = constrained_model->execute(input);
            std::cout << "Constrained inference completed successfully" << std::endl;
            
        } catch (const std::exception& e) {
            std::cout << "Constrained deployment failed: " << e.what() << std::endl;
        }
        
        // Example 3: Hardware recommendation
        std::cout << "\nExample 3: Hardware recommendation" << std::endl;
        try {
            auto recommended_hw = vmm.get_recommended_hardware("models/mobilenetv2.onnx");
            std::cout << "Recommended hardware: " << recommended_hw.get_name() << std::endl;
            
            // Test deployment on recommended hardware
            std::cout << "Testing deployment on recommended hardware..." << std::endl;
            auto recommended_model = vmm.deploy("models/mobilenetv2.onnx");
            std::cout << "Recommended hardware deployment successful!" << std::endl;
            
            // Test inference
            ai_vmm::Tensor input({1, 224, 224, 3}, ai_vmm::Precision::FP32);
            auto output = recommended_model->execute(input);
            std::cout << "Recommended hardware inference completed!" << std::endl;
            
        } catch (const std::exception& e) {
            std::cout << "Hardware recommendation test failed: " << e.what() << std::endl;
        }
        
        std::cout << "\nBasic usage example completed!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}