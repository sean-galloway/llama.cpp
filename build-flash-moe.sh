#!/bin/bash
# Build script for llama.cpp with Flash-MoE staged loading support
# This applies all necessary patches and builds the project

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="${SCRIPT_DIR}/build"

echo "=== Building llama.cpp with Flash-MoE support ==="
echo ""

# Check if we're in the right directory
if [ ! -f "${SCRIPT_DIR}/src/llama-model.cpp" ]; then
    echo "Error: This script must be run from the llama.cpp directory"
    exit 1
fi

# Check for required patch files
echo "=== Checking patch files ==="
required_patches=(
    "src/llama-model-flash-moe-integration.patch"
    "src/llama-graph-flash-moe-integration.patch"
    "src/CMakeLists-flash-moe.patch"
)

for patch in "${required_patches[@]}"; do
    if [ ! -f "${SCRIPT_DIR}/${patch}" ]; then
        echo "Warning: Patch file ${patch} not found"
    else
        echo "Found: ${patch}"
    fi
done

# Copy the complete staged moe files
echo ""
echo "=== Copying Flash-MoE source files ==="
cp -v "${SCRIPT_DIR}/src/llama-staged-moe-complete.h" "${SCRIPT_DIR}/src/llama-staged-moe.h" 2>/dev/null || true
cp -v "${SCRIPT_DIR}/src/llama-staged-moe-complete.cpp" "${SCRIPT_DIR}/src/llama-staged-moe.cpp" 2>/dev/null || true

# Apply patches
echo ""
echo "=== Applying patches ==="

# Apply model loader patch
if [ -f "${SCRIPT_DIR}/src/llama-model-flash-moe-integration.patch" ]; then
    echo "Applying model loader patch..."
    cd "${SCRIPT_DIR}"
    patch -p1 --forward --dry-run < "src/llama-model-flash-moe-integration.patch" 2>/dev/null && \
    patch -p1 --forward < "src/llama-model-flash-moe-integration.patch" || \
    echo "Model patch already applied or failed"
fi

# Apply graph patch
if [ -f "${SCRIPT_DIR}/src/llama-graph-flash-moe-integration.patch" ]; then
    echo "Applying graph patch..."
    cd "${SCRIPT_DIR}"
    patch -p1 --forward --dry-run < "src/llama-graph-flash-moe-integration.patch" 2>/dev/null && \
    patch -p1 --forward < "src/llama-graph-flash-moe-integration.patch" || \
    echo "Graph patch already applied or failed"
fi

# Apply CMake patch
if [ -f "${SCRIPT_DIR}/src/CMakeLists-flash-moe.patch" ]; then
    echo "Applying CMake patch..."
    cd "${SCRIPT_DIR}"
    patch -p1 --forward --dry-run < "src/CMakeLists-flash-moe.patch" 2>/dev/null && \
    patch -p1 --forward < "src/CMakeLists-flash-moe.patch" || \
    echo "CMake patch already applied or failed"
fi

# Create build directory
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

# Configure with CUDA support
echo ""
echo "=== Configuring build with CUDA ==="
cmake .. \
    -DLLAMA_CUDA=ON \
    -DLLAMA_NATIVE=ON \
    -DCMAKE_BUILD_TYPE=Release

# Build
echo ""
echo "=== Building llama-server ==="
make -j$(nproc) llama-server

echo ""
echo "=== Build complete ==="
echo ""
echo "To run with Flash-MoE staged loading:"
echo ""
echo "  export LLAMA_FLASH_MOE=1"
echo "  export LLAMA_FLASH_MOE_K=8"
echo "  ./build/bin/llama-server -m model.gguf -ngl 35"
echo ""
echo "For more information, see FLASH_MOE_USAGE.md"
