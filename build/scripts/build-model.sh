#!/bin/bash
# ==============================================================================
# Smith Model Build Script
# ==============================================================================
#
# DESCRIPTION:
#   Builds a contact model from the smith-models directory.
#   Works on both macOS (Docker) and LLNL HPC (Singularity/native).
#
# USAGE:
#   ./build-model.sh <model-name> [OPTIONS]
#
# ARGUMENTS:
#   model-name          Name of the model to build (required)
#
# OPTIONS:
#   -e, --env ENV       Build environment: docker, singularity, native (auto-detect)
#   -j, --jobs N        Number of parallel build jobs (default: auto-detect)
#   -c, --clean         Clean build directory before building
#   -v, --verbose       Verbose build output
#   -h, --help          Show this help message
#
# AVAILABLE MODELS:
#   From Puso & Laursen (2003):
#     - die-on-slab
#     - block-on-slab
#     - sphere-in-sphere
#
#   From Zimmerman & Ateshian (2018):
#     - stacked-blocks
#     - hemisphere-twisting
#     - concentric-spheres
#     - deep-indentation
#     - hollow-sphere-pinching
#
# EXAMPLES:
#   # Build die-on-slab model
#   ./build-model.sh die-on-slab
#
#   # Build with 8 jobs, clean first
#   ./build-model.sh sphere-in-sphere -j 8 --clean
#
#   # Force docker environment
#   ./build-model.sh block-on-slab --env docker
#
# PREREQUISITES:
#   Smith must be built first. Run one of:
#     - build/docker/build-smith-macos.sh    (macOS)
#     - build/hpc/build-smith-llnl.sh        (LLNL HPC)
#
# OUTPUT:
#   Build directory:  ../../smith-models/build/<model-name>/
#   Executable:       ../../smith-models/build/<model-name>/<model_name>
#
# ==============================================================================

set -e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default configuration
BUILD_ENV="auto"
BUILD_JOBS="auto"
CLEAN_BUILD=false
VERBOSE=false

# Help message
show_help() {
    grep '^#' "$0" | grep -v '#!/bin/bash' | sed 's/^# //g; s/^#//g'
    exit 0
}

# Check if model name is provided
if [[ $# -eq 0 ]]; then
    echo -e "${RED}Error: No model specified${NC}"
    echo "Usage: $0 <model-name> [OPTIONS]"
    echo
    echo "Available models:"
    echo "  die-on-slab, block-on-slab, sphere-in-sphere,"
    echo "  stacked-blocks, hemisphere-twisting, concentric-spheres,"
    echo "  deep-indentation, hollow-sphere-pinching"
    echo
    echo "Use --help for full documentation"
    exit 1
fi

MODEL_NAME="$1"
shift

# Parse options
while [[ $# -gt 0 ]]; do
    case $1 in
        -e|--env)
            BUILD_ENV="$2"
            shift 2
            ;;
        -j|--jobs)
            BUILD_JOBS="$2"
            shift 2
            ;;
        -c|--clean)
            CLEAN_BUILD=true
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo -e "${RED}Error: Unknown option $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
MODELS_DIR="${REPO_ROOT}/smith-models"
MODEL_SOURCE_DIR="${MODELS_DIR}/${MODEL_NAME}"
MODEL_BUILD_DIR="${MODELS_DIR}/build/${MODEL_NAME}"
SMITH_INSTALL_DIR="${REPO_ROOT}/smith-install"

# Validate model exists
if [[ ! -d "${MODEL_SOURCE_DIR}" ]]; then
    echo -e "${RED}Error: Model '${MODEL_NAME}' not found${NC}"
    echo "Expected directory: ${MODEL_SOURCE_DIR}"
    echo
    echo "Available models in ${MODELS_DIR}:"
    ls -1 "${MODELS_DIR}" | grep -v README.md | grep -v build || echo "  (none found)"
    exit 1
fi

if [[ ! -f "${MODEL_SOURCE_DIR}/CMakeLists.txt" ]]; then
    echo -e "${RED}Error: No CMakeLists.txt found for model '${MODEL_NAME}'${NC}"
    exit 1
fi

# Check if Smith is built
if [[ ! -d "${SMITH_INSTALL_DIR}" ]]; then
    echo -e "${RED}Error: Smith not found. Please build Smith first.${NC}"
    echo
    echo "To build Smith:"
    echo "  macOS:     build/docker/build-smith-macos.sh"
    echo "  LLNL HPC:  build/hpc/build-smith-llnl.sh"
    exit 1
fi

SMITH_CMAKE_DIR="${SMITH_INSTALL_DIR}/lib/cmake"
if [[ ! -d "${SMITH_CMAKE_DIR}" ]]; then
    echo -e "${RED}Error: Smith CMake config not found: ${SMITH_CMAKE_DIR}${NC}"
    echo "Smith may not be properly installed."
    exit 1
fi

# Auto-detect build environment
if [[ "$BUILD_ENV" == "auto" ]]; then
    if command -v docker &> /dev/null && docker info &> /dev/null 2>&1; then
        BUILD_ENV="docker"
    elif command -v singularity &> /dev/null; then
        BUILD_ENV="singularity"
    else
        BUILD_ENV="native"
    fi
    echo -e "${YELLOW}Auto-detected build environment: ${BUILD_ENV}${NC}"
fi

# Auto-detect build jobs
if [[ "$BUILD_JOBS" == "auto" ]]; then
    if command -v nproc &> /dev/null; then
        BUILD_JOBS=$(nproc)
    elif [[ -f /proc/cpuinfo ]]; then
        BUILD_JOBS=$(grep -c ^processor /proc/cpuinfo)
    else
        BUILD_JOBS=4
    fi
fi

# Print configuration
echo -e "${BLUE}=== Building Smith Model ===${NC}"
echo
echo "Configuration:"
echo "  Model:            ${MODEL_NAME}"
echo "  Source Directory: ${MODEL_SOURCE_DIR}"
echo "  Build Directory:  ${MODEL_BUILD_DIR}"
echo "  Smith Install:    ${SMITH_INSTALL_DIR}"
echo "  Build Environment: ${BUILD_ENV}"
echo "  Build Jobs:       ${BUILD_JOBS}"
echo "  Clean Build:      ${CLEAN_BUILD}"
echo

# Clean if requested
if [[ "$CLEAN_BUILD" == "true" && -d "${MODEL_BUILD_DIR}" ]]; then
    echo -e "${YELLOW}Cleaning build directory...${NC}"
    rm -rf "${MODEL_BUILD_DIR}"
fi

# Create build directory
mkdir -p "${MODEL_BUILD_DIR}"

# Build based on environment
case "$BUILD_ENV" in
    docker)
        # Docker build (macOS)
        DOCKER_IMAGE="seracllnl/tpls:clang-19_10-09-25_23h-54m"
        HOST_CONFIG="llvm@19.1.1.cmake"

        if ! docker image inspect "${DOCKER_IMAGE}" > /dev/null 2>&1; then
            echo -e "${RED}Error: Docker image not found: ${DOCKER_IMAGE}${NC}"
            echo "Please run build/docker/build-smith-macos.sh first"
            exit 1
        fi

        VERBOSE_FLAG=""
        if [[ "$VERBOSE" == "true" ]]; then
            VERBOSE_FLAG="-DCMAKE_VERBOSE_MAKEFILE=ON"
        fi

        echo -e "${GREEN}Building model in Docker...${NC}"
        docker run --rm \
            --platform linux/amd64 \
            -u serac \
            -v "${MODEL_SOURCE_DIR}:/workspace/model:ro" \
            -v "${MODEL_BUILD_DIR}:/workspace/build" \
            -v "${SMITH_INSTALL_DIR}:/smith-install:ro" \
            -w /workspace/build \
            "${DOCKER_IMAGE}" \
            /bin/bash -c "
                set -e
                echo 'Configuring model...'
                cmake /workspace/model \
                    -C /home/serac/tpls/host-configs/docker/${HOST_CONFIG} \
                    -DSmith_DIR=/smith-install/lib/cmake \
                    -DCMAKE_BUILD_TYPE=Release \
                    ${VERBOSE_FLAG}

                echo ''
                echo 'Building model...'
                make -j${BUILD_JOBS} VERBOSE=${VERBOSE}
            "
        ;;

    singularity)
        # Singularity build (LLNL HPC)
        CONTAINER_IMAGE="${HOME}/containers/smith-clang19.sif"
        HOST_CONFIG="llvm@19.1.1.cmake"

        if [[ ! -f "${CONTAINER_IMAGE}" ]]; then
            echo -e "${RED}Error: Singularity container not found: ${CONTAINER_IMAGE}${NC}"
            echo "Please run build/hpc/build-smith-llnl.sh first"
            exit 1
        fi

        VERBOSE_FLAG=""
        if [[ "$VERBOSE" == "true" ]]; then
            VERBOSE_FLAG="-DCMAKE_VERBOSE_MAKEFILE=ON"
        fi

        echo -e "${GREEN}Building model with Singularity...${NC}"
        singularity exec \
            --bind ${MODEL_SOURCE_DIR}:/workspace/model:ro \
            --bind ${MODEL_BUILD_DIR}:/workspace/build \
            --bind ${SMITH_INSTALL_DIR}:/smith-install:ro \
            ${CONTAINER_IMAGE} \
            /bin/bash -c "
                set -e
                cd /workspace/build
                echo 'Configuring model...'
                cmake /workspace/model \
                    -C /home/serac/tpls/host-configs/docker/${HOST_CONFIG} \
                    -DSmith_DIR=/smith-install/lib/cmake \
                    -DCMAKE_BUILD_TYPE=Release \
                    ${VERBOSE_FLAG}

                echo ''
                echo 'Building model...'
                make -j${BUILD_JOBS} VERBOSE=${VERBOSE}
            "
        ;;

    native)
        # Native build
        VERBOSE_FLAG=""
        if [[ "$VERBOSE" == "true" ]]; then
            VERBOSE_FLAG="-DCMAKE_VERBOSE_MAKEFILE=ON"
        fi

        echo -e "${GREEN}Building model natively...${NC}"
        cd "${MODEL_BUILD_DIR}"

        echo "Configuring model..."
        cmake "${MODEL_SOURCE_DIR}" \
            -DSmith_DIR="${SMITH_CMAKE_DIR}" \
            -DCMAKE_BUILD_TYPE=Release \
            ${VERBOSE_FLAG}

        echo
        echo "Building model..."
        if [[ "$VERBOSE" == "true" ]]; then
            make -j${BUILD_JOBS} VERBOSE=1
        else
            make -j${BUILD_JOBS}
        fi
        ;;

    *)
        echo -e "${RED}Error: Unknown build environment '${BUILD_ENV}'${NC}"
        echo "Available environments: docker, singularity, native"
        exit 1
        ;;
esac

# Build status
if [[ $? -eq 0 ]]; then
    # Find the executable
    MODEL_EXECUTABLE=$(find "${MODEL_BUILD_DIR}" -maxdepth 1 -type f -executable 2>/dev/null | head -1)

    echo
    echo -e "${GREEN}✓ Model build complete!${NC}"
    echo
    echo "Build output:"
    echo "  Directory: ${MODEL_BUILD_DIR}"
    if [[ -n "${MODEL_EXECUTABLE}" ]]; then
        echo "  Executable: ${MODEL_EXECUTABLE}"
        echo
        echo "Next steps:"
        echo "  Run model: build/scripts/run-model.sh ${MODEL_NAME}"
    fi
else
    echo
    echo -e "${RED}✗ Model build failed${NC}"
    exit 1
fi
