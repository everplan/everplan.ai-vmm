"""
AI VMM Python Bindings

This module provides Python bindings for the AI Virtual Machine Manager,
enabling easy deployment of AI models across heterogeneous hardware.

Example:
    import ai_vmm
    
    # Create VMM instance
    vmm = ai_vmm.VMM()
    
    # Deploy a model
    model = vmm.deploy("path/to/model.onnx")
    
    # Execute inference
    import numpy as np
    input_data = np.random.rand(1, 224, 224, 3).astype(np.float32)
    output = model.execute(input_data)
"""

__version__ = "0.1.0"
__author__ = "AI VMM Team"

# Import main classes when available
try:
    from ._ai_vmm import VMM, Tensor, DeploymentConstraints, HardwareType, ModelCategory, Precision
    __all__ = ['VMM', 'Tensor', 'DeploymentConstraints', 'HardwareType', 'ModelCategory', 'Precision']
except ImportError:
    # Fallback for when C++ module is not built
    print("AI VMM C++ module not available. Please build the project first.")
    __all__ = []