#!/bin/bash
# Build script for Smith FEA package
# This script automates the build process using Spack/uberenv

set -e  # Exit on error

echo "================================"
echo "Smith FEA Build Script"
echo "================================"

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SMITH_DIR="${SCRIPT_DIR}/smith"
TPL_DIR="${SCRIPT_DIR}/smith_tpls"
BUILD_DIR="${SMITH_DIR}/build"
COMPILER="${COMPILER:-gcc_13}"
BUILD_SPEC="${BUILD_SPEC:-~devtools~enzyme %${COMPILER}}"
BUILD_JOBS="${BUILD_JOBS:-$(nproc)}"
# Allow override of spack config path
SPACK_CONFIG="${SPACK_CONFIG:-./scripts/spack/configs/docker/ubuntu24/spack.yaml}"

echo "Configuration:"
echo "  Smith directory: ${SMITH_DIR}"
echo "  TPL directory: ${TPL_DIR}"
echo "  Build directory: ${BUILD_DIR}"
echo "  Compiler: ${COMPILER}"
echo "  Build spec: ${BUILD_SPEC}"
echo "  Parallel jobs: ${BUILD_JOBS}"
echo "  Spack config: ${SPACK_CONFIG}"
echo ""

# Check prerequisites
echo "Checking prerequisites..."

if ! command -v cmake &> /dev/null; then
    echo "ERROR: cmake not found. Please install cmake 3.14+"
    exit 1
fi

if ! command -v python3 &> /dev/null; then
    echo "ERROR: python3 not found"
    exit 1
fi

if ! command -v mpicc &> /dev/null; then
    echo "ERROR: MPI not found. Please install mpich or openmpi"
    echo "  Ubuntu/Debian: apt-get install mpich libmpich-dev"
    exit 1
fi

if ! command -v gfortran &> /dev/null; then
    echo "ERROR: gfortran not found. Please install gfortran"
    echo "  Ubuntu/Debian: apt-get install gfortran"
    exit 1
fi

CMAKE_VERSION=$(cmake --version | head -1 | awk '{print $3}')
echo "  CMake version: ${CMAKE_VERSION}"
echo "  Python version: $(python3 --version)"
echo "  MPI: $(mpicc --version | head -1)"
echo "  GCC: $(gcc --version | head -1)"
echo "  gfortran: $(gfortran --version | head -1)"
echo ""

# Check if smith submodule is initialized
if [ ! -f "${SMITH_DIR}/CMakeLists.txt" ]; then
    echo "ERROR: Smith submodule not initialized"
    echo "Please run: git submodule update --init --recursive"
    exit 1
fi
echo "Smith submodule: initialized"
echo ""

# Step 1: Build third-party libraries with uberenv
echo "================================"
echo "Step 1: Building dependencies with uberenv"
echo "================================"

cd "${SMITH_DIR}"

if [ -d "${TPL_DIR}" ] && [ -n "$(ls -A ${TPL_DIR} 2>/dev/null)" ]; then
    echo "WARNING: TPL directory ${TPL_DIR} already exists"
    read -p "Remove and rebuild? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing TPL directory..."
        rm -rf "${TPL_DIR}"
    else
        echo "Using existing TPL directory"
    fi
fi

if [ ! -d "${TPL_DIR}/spack" ]; then
    echo "Running uberenv to build dependencies..."
    echo "NOTE: This requires network access to download Spack packages."
    echo "      In restricted environments, this step will fail."
    echo ""

    python3 ./scripts/uberenv/uberenv.py \
        --spack-env-file="${SPACK_CONFIG}" \
        --project-json=.uberenv_config.json \
        --spec="${BUILD_SPEC}" \
        --prefix="${TPL_DIR}" \
        -k \
        -j "${BUILD_JOBS}"

    if [ $? -ne 0 ]; then
        echo ""
        echo "================================"
        echo "ERROR: uberenv build failed"
        echo "================================"
        echo ""
        echo "This is likely due to network access restrictions."
        echo "See SMITH_BUILD_STATUS.md for alternative build approaches."
        echo ""
        exit 1
    fi
else
    echo "Skipping uberenv (TPLs already built)"
fi

echo "Dependencies built successfully"
echo ""

# Step 2: Configure Smith with generated host-config
echo "================================"
echo "Step 2: Configuring Smith"
echo "================================"

# Find the generated host-config file
HOSTCONFIG=$(ls -t ${SMITH_DIR}/*.cmake 2>/dev/null | grep -v CMakeLists | head -1)

if [ -z "${HOSTCONFIG}" ]; then
    echo "ERROR: No host-config file found"
    echo "Expected *.cmake file in ${SMITH_DIR}"
    exit 1
fi

echo "Using host-config: ${HOSTCONFIG}"

if [ -d "${BUILD_DIR}" ]; then
    echo "WARNING: Build directory ${BUILD_DIR} already exists"
    read -p "Remove and reconfigure? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "Removing existing build directory..."
        rm -rf "${BUILD_DIR}"
    fi
fi

python3 config-build.py -hc "${HOSTCONFIG}" -bp build

echo "Configuration complete"
echo ""

# Step 3: Build Smith
echo "================================"
echo "Step 3: Building Smith"
echo "================================"

cd "${BUILD_DIR}"

echo "Compiling with ${BUILD_JOBS} parallel jobs..."
make -j"${BUILD_JOBS}" VERBOSE=1

if [ $? -ne 0 ]; then
    echo "ERROR: Build failed"
    exit 1
fi

echo "Build complete"
echo ""

# Step 4: Run tests
echo "================================"
echo "Step 4: Running tests"
echo "================================"

make test ARGS="-VV" -j"${BUILD_JOBS}"

if [ $? -ne 0 ]; then
    echo "WARNING: Some tests failed"
else
    echo "All tests passed"
fi

echo ""
echo "================================"
echo "Build Complete!"
echo "================================"
echo "Smith executable and libraries are in: ${BUILD_DIR}"
echo ""
echo "Example usage:"
echo "  cd ${BUILD_DIR}"
echo "  mpirun -np 1 ./examples/example_name"
echo ""
