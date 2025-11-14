#!/bin/bash

# AI VMM Build Script
# Builds the project with different configurations

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 AI VMM Build Script${NC}"
echo "======================================="

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "CMakeLists.txt" ]; then
    print_error "CMakeLists.txt not found. Please run from project root."
    exit 1
fi

# Parse command line arguments
BUILD_TYPE="Release"
ENABLE_INTEL="ON"
ENABLE_NVIDIA="OFF"  # Default off since CUDA may not be available
ENABLE_AMD="OFF"
BUILD_TESTS="ON"
BUILD_PYTHON="OFF"   # Default off since pybind11 may not be available
CLEAN_BUILD="false"

while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --clean)
            CLEAN_BUILD="true"
            shift
            ;;
        --nvidia)
            ENABLE_NVIDIA="ON"
            shift
            ;;
        --no-intel)
            ENABLE_INTEL="OFF"
            shift
            ;;
        --python)
            BUILD_PYTHON="ON"
            shift
            ;;
        --no-tests)
            BUILD_TESTS="OFF"
            shift
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo "Options:"
            echo "  --debug       Build in Debug mode (default: Release)"
            echo "  --clean       Clean build directory before building"
            echo "  --nvidia      Enable NVIDIA backend (requires CUDA)"
            echo "  --no-intel    Disable Intel backend"
            echo "  --python      Enable Python bindings (requires pybind11)"
            echo "  --no-tests    Disable building tests"
            echo "  -h, --help    Show this help message"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Create build directory
BUILD_DIR="build"
if [ "$CLEAN_BUILD" = "true" ] && [ -d "$BUILD_DIR" ]; then
    print_status "Cleaning build directory..."
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

print_status "Configuring CMake..."
print_status "Build Type: $BUILD_TYPE"
print_status "Intel Backend: $ENABLE_INTEL"
print_status "NVIDIA Backend: $ENABLE_NVIDIA"
print_status "AMD Backend: $ENABLE_AMD"
print_status "Build Tests: $BUILD_TESTS"
print_status "Python Bindings: $BUILD_PYTHON"

# Configure with CMake
CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
    -DENABLE_INTEL_BACKEND="$ENABLE_INTEL"
    -DENABLE_NVIDIA_BACKEND="$ENABLE_NVIDIA"
    -DENABLE_AMD_BACKEND="$ENABLE_AMD"
    -DBUILD_TESTS="$BUILD_TESTS"
    -DBUILD_PYTHON_BINDINGS="$BUILD_PYTHON"
)

if ! cmake "${CMAKE_ARGS[@]}" ..; then
    print_error "CMake configuration failed"
    exit 1
fi

print_status "Building project..."

# Determine number of cores for parallel build
CORES=$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

if ! make -j"$CORES"; then
    print_error "Build failed"
    exit 1
fi

print_status "Build completed successfully!"

# Run tests if enabled
if [ "$BUILD_TESTS" = "ON" ]; then
    print_status "Running tests..."
    if ! ctest --output-on-failure; then
        print_warning "Some tests failed"
    else
        print_status "All tests passed!"
    fi
fi

# Build examples
if [ -f "examples/basic_usage/ai_vmm_basic_example" ]; then
    print_status "Running basic example..."
    if ./examples/basic_usage/ai_vmm_basic_example; then
        print_status "Basic example completed successfully!"
    else
        print_warning "Basic example had issues (expected in skeleton implementation)"
    fi
fi

echo
print_status "Build artifacts location: $(pwd)"
print_status "To install, run: sudo make install"

if [ "$BUILD_PYTHON" = "ON" ]; then
    print_status "To install Python package, run: pip install -e .."
fi

echo -e "${GREEN}✅ AI VMM build completed successfully!${NC}"