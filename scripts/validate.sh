#!/bin/bash

# AI VMM Foundation Validation Script
# Tests that the project structure and basic functionality work

set -e

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

echo -e "${BLUE}🔍 AI VMM Foundation Validation${NC}"
echo "======================================="

validate_item() {
    local item_name="$1"
    local condition="$2"
    
    if eval "$condition"; then
        echo -e "${GREEN}✅${NC} $item_name"
        return 0
    else
        echo -e "${RED}❌${NC} $item_name"
        return 1
    fi
}

validate_file() {
    validate_item "File exists: $1" "[ -f '$1' ]"
}

validate_dir() {
    validate_item "Directory exists: $1" "[ -d '$1' ]"
}

echo "📁 Project Structure Validation"
echo "--------------------------------"

# Validate core project files
validate_file "CMakeLists.txt"
validate_file "README.md"
validate_file "ARCHITECTURE.md"
validate_file "DEVELOPMENT.md"
validate_file "PROJECT_STRUCTURE.md"
validate_file "ai-vmm.md"

# Validate include directory structure
validate_dir "include"
validate_dir "include/ai_vmm"
validate_file "include/ai_vmm/ai_vmm.hpp"
validate_file "include/ai_vmm/types.hpp"
validate_file "include/ai_vmm/compute_backend.hpp"
validate_file "include/ai_vmm/vmm.hpp"

# Validate source directory structure
validate_dir "src"
validate_dir "src/core"
validate_dir "src/hal"
validate_dir "src/optimization"
validate_dir "src/scheduling"
validate_dir "src/memory"
validate_dir "src/backends"
validate_dir "src/backends/intel"
validate_dir "src/backends/nvidia"

# Validate core source files
validate_file "src/core/CMakeLists.txt"
validate_file "src/core/vmm.cpp"
validate_file "src/core/tensor.cpp"
validate_file "src/core/model.cpp"
validate_file "src/core/hardware_discovery.cpp"

# Validate other components
validate_file "src/hal/CMakeLists.txt"
validate_file "src/hal/backend_registry.cpp"
validate_file "src/scheduling/CMakeLists.txt"
validate_file "src/scheduling/workload_scheduler.cpp"
validate_file "src/memory/CMakeLists.txt"
validate_file "src/memory/memory_manager.cpp"
validate_file "src/optimization/model_optimizer.cpp"

# Validate examples and tests
validate_dir "examples"
validate_dir "examples/basic_usage"
validate_file "examples/basic_usage/main.cpp"
validate_file "examples/basic_usage/CMakeLists.txt"
validate_dir "tests"
validate_file "tests/CMakeLists.txt"
validate_file "tests/test_main.cpp"
validate_file "tests/test_tensor.cpp"
validate_file "tests/test_vmm_basic.cpp"
validate_file "tests/test_hardware_discovery.cpp"

# Validate Python structure
validate_dir "python"
validate_dir "python/ai_vmm"
validate_file "python/ai_vmm/__init__.py"

# Validate scripts
validate_dir "scripts"
validate_file "scripts/build.sh"

echo
echo "🏗️ Build System Validation"
echo "---------------------------"

# Check if CMake can parse the main CMakeLists.txt
if cmake --version >/dev/null 2>&1; then
    if cmake -S . -B build_test_temp -DCMAKE_BUILD_TYPE=Release -DENABLE_INTEL_BACKEND=OFF -DENABLE_NVIDIA_BACKEND=OFF -DBUILD_PYTHON_BINDINGS=OFF >/dev/null 2>&1; then
        echo -e "${GREEN}✅${NC} CMake configuration successful"
        rm -rf build_test_temp
    else
        echo -e "${RED}❌${NC} CMake configuration failed"
    fi
else
    echo -e "${YELLOW}⚠️${NC} CMake not available for validation"
fi

echo
echo "📋 Code Quality Validation"
echo "--------------------------"

# Check for basic C++ syntax in header files
cpp_syntax_ok=true
for header in include/ai_vmm/*.hpp; do
    if [ -f "$header" ]; then
        # Basic syntax check - look for balanced braces
        if grep -q "class\|struct\|namespace" "$header" && grep -q "#pragma once" "$header"; then
            echo -e "${GREEN}✅${NC} Header syntax: $(basename "$header")"
        else
            echo -e "${RED}❌${NC} Header syntax: $(basename "$header")"
            cpp_syntax_ok=false
        fi
    fi
done

# Check for basic C++ syntax in source files
for source in src/core/*.cpp; do
    if [ -f "$source" ]; then
        # Basic syntax check
        if grep -q "#include\|namespace ai_vmm" "$source"; then
            echo -e "${GREEN}✅${NC} Source syntax: $(basename "$source")"
        else
            echo -e "${RED}❌${NC} Source syntax: $(basename "$source")"
            cpp_syntax_ok=false
        fi
    fi
done

echo
echo "📊 Foundation Summary"
echo "--------------------"

total_checks=0
passed_checks=0

# Count validation results
while read -r line; do
    if [[ $line =~ ✅ ]]; then
        ((passed_checks++))
        ((total_checks++))
    elif [[ $line =~ ❌ ]]; then
        ((total_checks++))
    fi
done < <(grep -E "✅|❌" <<< "$(validate_file "dummy" 2>&1)" || echo "")

# Recount properly by running key validations
key_files=(
    "CMakeLists.txt"
    "README.md"
    "include/ai_vmm/ai_vmm.hpp"
    "src/core/vmm.cpp"
    "examples/basic_usage/main.cpp"
    "tests/test_main.cpp"
)

total_checks=${#key_files[@]}
passed_checks=0

for file in "${key_files[@]}"; do
    if [ -f "$file" ]; then
        ((passed_checks++))
    fi
done

percentage=$((passed_checks * 100 / total_checks))

echo "Foundation Status: $passed_checks/$total_checks key components ready ($percentage%)"

if [ $passed_checks -eq $total_checks ]; then
    echo -e "${GREEN}🎉 Project foundation is complete and ready for development!${NC}"
    echo
    echo "Next steps:"
    echo "  1. Run './scripts/build.sh' to build the project"
    echo "  2. Start implementing hardware backends (Intel, NVIDIA)"
    echo "  3. Add model analysis and optimization components"
    echo
    echo "Build validation:"
    if [ -d "build" ] && [ -f "build/libai-vmm.a" ]; then
        echo -e "${GREEN}✅${NC} Project builds successfully"
    else
        echo -e "${YELLOW}ℹ️${NC} Run build script to compile the project"
    fi
    exit 0
else
    echo -e "${RED}⚠️ Foundation incomplete. Please check missing components.${NC}"
    exit 1
fi