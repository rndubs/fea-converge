#!/bin/bash
# ==============================================================================
# Smith Docker Build Script for macOS
# ==============================================================================
#
# DESCRIPTION:
#   Builds Smith using Docker containers with pre-built TPLs from seracllnl.
#   Designed for macOS (both Intel and Apple Silicon).
#
# USAGE:
#   ./build-smith-macos.sh [OPTIONS]
#
# OPTIONS:
#   -i, --image IMAGE     Docker image to use (default: clang-19)
#   -c, --config CONFIG   Host config file (default: llvm@19.1.1.cmake)
#   -j, --jobs N          Number of parallel build jobs (default: auto-detect)
#   -t, --test            Run tests after building
#   -h, --help            Show this help message
#
# AVAILABLE IMAGES:
#   clang-19  : seracllnl/tpls:clang-19_10-09-25_23h-54m (recommended)
#   gcc-14    : seracllnl/tpls:gcc-14_10-09-25_23h-54m
#   cuda-12   : seracllnl/tpls:cuda-12_04-16-25_20h-55m (requires NVIDIA GPU)
#
# EXAMPLES:
#   # Build with default settings (Clang 19)
#   ./build-smith-macos.sh
#
#   # Build with GCC 14
#   ./build-smith-macos.sh --image gcc-14
#
#   # Build with 8 jobs and run tests
#   ./build-smith-macos.sh -j 8 --test
#
# OUTPUT:
#   Build artifacts:   ../../smith-build/
#   Install location:  ../../smith-install/
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
DOCKER_IMAGE_NAME="clang-19"
BUILD_JOBS="auto"
RUN_TESTS=false

# Image mappings
declare -A IMAGE_MAP=(
    ["clang-19"]="seracllnl/tpls:clang-19_10-09-25_23h-54m"
    ["gcc-14"]="seracllnl/tpls:gcc-14_10-09-25_23h-54m"
    ["cuda-12"]="seracllnl/tpls:cuda-12_04-16-25_20h-55m"
)

declare -A CONFIG_MAP=(
    ["clang-19"]="llvm@19.1.1.cmake"
    ["gcc-14"]="gcc@14.2.0.cmake"
    ["cuda-12"]="gcc@12.3.0_cuda.cmake"
)

# Help message
show_help() {
    grep '^#' "$0" | grep -v '#!/bin/bash' | sed 's/^# //g; s/^#//g'
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -i|--image)
            DOCKER_IMAGE_NAME="$2"
            shift 2
            ;;
        -c|--config)
            CUSTOM_CONFIG="$2"
            shift 2
            ;;
        -j|--jobs)
            BUILD_JOBS="$2"
            shift 2
            ;;
        -t|--test)
            RUN_TESTS=true
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

# Validate image name
if [[ -z "${IMAGE_MAP[$DOCKER_IMAGE_NAME]}" ]]; then
    echo -e "${RED}Error: Unknown image name '$DOCKER_IMAGE_NAME'${NC}"
    echo "Available images: ${!IMAGE_MAP[@]}"
    exit 1
fi

DOCKER_IMAGE="${IMAGE_MAP[$DOCKER_IMAGE_NAME]}"
HOST_CONFIG="${CUSTOM_CONFIG:-${CONFIG_MAP[$DOCKER_IMAGE_NAME]}}"

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SMITH_DIR="${REPO_ROOT}/smith"
BUILD_DIR="${REPO_ROOT}/smith-build"
INSTALL_DIR="${REPO_ROOT}/smith-install"

# Print configuration
echo -e "${BLUE}=== Smith Docker Build for macOS ===${NC}"
echo
echo "Configuration:"
echo "  Docker Image:     ${DOCKER_IMAGE}"
echo "  Host Config:      ${HOST_CONFIG}"
echo "  Smith Source:     ${SMITH_DIR}"
echo "  Build Directory:  ${BUILD_DIR}"
echo "  Install Directory: ${INSTALL_DIR}"
echo "  Build Jobs:       ${BUILD_JOBS}"
echo "  Run Tests:        ${RUN_TESTS}"
echo

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}Error: Docker is not running.${NC}"
    echo "Please start Docker Desktop and try again."
    exit 1
fi

# Check if Smith submodule exists
if [[ ! -f "${SMITH_DIR}/CMakeLists.txt" ]]; then
    echo -e "${RED}Error: Smith submodule not found.${NC}"
    echo "Please run: git submodule update --init --recursive"
    exit 1
fi

# Create build directories
mkdir -p "${BUILD_DIR}"
mkdir -p "${INSTALL_DIR}"

# Check if image exists locally
if ! docker image inspect "${DOCKER_IMAGE}" > /dev/null 2>&1; then
    echo -e "${YELLOW}Docker image not found locally. Pulling ${DOCKER_IMAGE}...${NC}"
    echo -e "${YELLOW}(This is ~4 GB and may take several minutes)${NC}"
    if [[ $(uname -m) == "arm64" ]]; then
        echo -e "${YELLOW}Note: Running x86_64 image on Apple Silicon via Rosetta 2${NC}"
    fi
    docker pull --platform linux/amd64 "${DOCKER_IMAGE}"
else
    echo -e "${GREEN}✓ Docker image found locally${NC}"
fi

echo
echo -e "${GREEN}Building Smith...${NC}"
echo

# Build command
BUILD_CMD="set -e
echo '=== Configuring Smith ==='
cd /home/serac/serac
python3 ./config-build.py \
    -hc host-configs/docker/${HOST_CONFIG} \
    -bp /home/serac/smith-build \
    -ip /home/serac/smith-install

echo ''
echo '=== Building Smith ==='
cd /home/serac/smith-build
"

if [[ "$BUILD_JOBS" == "auto" ]]; then
    BUILD_CMD+="make -j\$(nproc)"
else
    BUILD_CMD+="make -j${BUILD_JOBS}"
fi

BUILD_CMD+="
echo ''
echo '=== Installing Smith ==='
make install
"

if [[ "$RUN_TESTS" == "true" ]]; then
    BUILD_CMD+="
echo ''
echo '=== Running Tests ==='
ctest --output-on-failure -j\$(nproc)
"
fi

# Run the build
docker run --rm \
    --platform linux/amd64 \
    -u serac \
    -v "${SMITH_DIR}:/home/serac/serac:ro" \
    -v "${BUILD_DIR}:/home/serac/smith-build" \
    -v "${INSTALL_DIR}:/home/serac/smith-install" \
    "${DOCKER_IMAGE}" \
    /bin/bash -c "${BUILD_CMD}"

# Build status
if [[ $? -eq 0 ]]; then
    echo
    echo -e "${GREEN}✓ Smith build complete!${NC}"
    echo
    echo "Build artifacts:"
    echo "  Build:   ${BUILD_DIR}"
    echo "  Install: ${INSTALL_DIR}"
    echo
    echo "Next steps:"
    echo "  - Build models: cd build/scripts && ./build-model.sh <model-name>"
    echo "  - Run models:   cd build/scripts && ./run-model.sh <model-name>"
else
    echo
    echo -e "${RED}✗ Smith build failed${NC}"
    exit 1
fi
