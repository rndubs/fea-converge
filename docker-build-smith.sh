#!/bin/bash
# ==============================================================================
# Smith Docker Build Script for macOS
# ==============================================================================
#
# OVERVIEW:
# This script provides a Docker-based build environment for Smith on macOS.
# It uses pre-built Docker images from the Serac project that contain all
# third-party libraries (TPLs) already compiled, avoiding the network access
# limitations and long compile times of direct builds.
#
# QUICK START:
#   ./docker-build-smith.sh
#
# This will:
#   1. Pull the latest Clang 19 Docker image (~4 GB, one-time download)
#   2. Mount your local ./smith directory into the container
#   3. Start an interactive shell where you can build Smith
#
# INSIDE THE CONTAINER:
#   cd /home/serac/serac
#   python3 ./config-build.py -hc host-configs/docker/llvm@19.1.1.cmake -bp ../smith-build -ip ../smith-install
#   cd ../smith-build
#   make -j$(nproc)
#   make test
#
# AVAILABLE DOCKER IMAGES (October 2025):
#
# 1. Clang 19 (RECOMMENDED for macOS):
#    export DOCKER_IMAGE=seracllnl/tpls:clang-19_10-09-25_23h-54m
#    export HOST_CONFIG=llvm@19.1.1.cmake
#    Features: Clang 19.1.1, LLVM toolchain, dev tools (clang-format, clang-tidy, docs)
#
# 2. GCC 14 (Alternative):
#    export DOCKER_IMAGE=seracllnl/tpls:gcc-14_10-09-25_23h-54m
#    export HOST_CONFIG=gcc@14.2.0.cmake
#    Features: GCC 14.2.0, good for testing compiler compatibility
#
# 3. CUDA 12 (GPU Computing - requires NVIDIA GPU):
#    export DOCKER_IMAGE=seracllnl/tpls:cuda-12_04-16-25_20h-55m
#    export HOST_CONFIG=gcc@12.3.0_cuda.cmake
#    Features: CUDA 12, GCC 12.3.0, GPU-accelerated libraries
#
# VOLUME MOUNTING:
# The -v flag mounts your local ./smith directory at /home/serac/serac in the container.
# - Changes inside the container are reflected on your macOS filesystem
# - You can edit code on macOS and build inside the container
# - Build artifacts (../smith-build) remain in the container's filesystem
#
# MACOS ARM64 (Apple Silicon) COMPATIBILITY:
# Docker images are x86_64. On M1/M2/M3 Macs, Docker uses Rosetta 2 emulation.
# This works correctly but may be slower than native ARM builds.
#
# TROUBLESHOOTING:
# - "Docker not running": Launch Docker Desktop from Applications
# - "No space left": Docker images are ~4 GB each. Run: docker system prune -a
# - "Permission denied": Script uses -u serac for correct permissions
#
# ADVANCED USAGE:
# - Custom CMake options: Add after -- in config-build.py command
#   python ./config-build.py -hc ... -- -DBUILD_SHARED_LIBS=ON
# - Parallel builds: make -j8 (use 8 cores)
# - Specific tests: ctest -R test_name
# - Verbose tests: ctest -V
# - Copy artifacts: docker cp <container-id>:/home/serac/smith-build/bin ./bin
#
# REFERENCES:
# - Serac Docs: https://serac.readthedocs.io/en/latest/sphinx/dev_guide/docker_env.html
# - Docker Hub: https://hub.docker.com/r/seracllnl/tpls/tags
# - Smith CI: ./smith/.github/workflows/ci-tests.yml (shows latest tested images)
#
# ==============================================================================

set -e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Smith Docker Build Setup ===${NC}"
echo

# Configuration - Override these with environment variables
SMITH_REPO_PATH="$(cd "$(dirname "$0")" && pwd)"
DOCKER_IMAGE="${DOCKER_IMAGE:-seracllnl/tpls:clang-19_10-09-25_23h-54m}"
HOST_CONFIG="${HOST_CONFIG:-llvm@19.1.1.cmake}"
BUILD_DIR="${BUILD_DIR:-../smith-build}"
INSTALL_DIR="${INSTALL_DIR:-../smith-install}"

echo "Configuration:"
echo "  Docker Image: ${DOCKER_IMAGE}"
echo "  Host Config:  ${HOST_CONFIG}"
echo "  Smith Repo:   ${SMITH_REPO_PATH}/smith"
echo "  Build Dir:    ${BUILD_DIR}"
echo "  Install Dir:  ${INSTALL_DIR}"
echo

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}Error: Docker is not running. Please start Docker Desktop and try again.${NC}"
    exit 1
fi

# Check if image exists locally, if not pull it
if ! docker image inspect "${DOCKER_IMAGE}" > /dev/null 2>&1; then
    echo -e "${YELLOW}Docker image not found locally. Pulling ${DOCKER_IMAGE}...${NC}"
    echo -e "${YELLOW}(This is ~4 GB and may take several minutes)${NC}"
    echo -e "${YELLOW}Note: Using x86_64 emulation on Apple Silicon via Rosetta 2${NC}"
    docker pull --platform linux/amd64 "${DOCKER_IMAGE}"
else
    echo -e "${GREEN}Docker image ${DOCKER_IMAGE} found locally.${NC}"
fi

echo
echo -e "${GREEN}Starting Docker container and building Smith...${NC}"
echo

# Detect if running interactively or in a script
if [ -t 0 ]; then
    # Interactive mode - drop into shell
    echo "Interactive mode: Starting shell inside container"
    echo
    echo "Inside the container, run these commands to build Smith:"
    echo
    echo -e "${YELLOW}  cd /home/serac/serac${NC}"
    echo -e "${YELLOW}  python3 ./config-build.py -hc host-configs/docker/${HOST_CONFIG} -bp ${BUILD_DIR} -ip ${INSTALL_DIR}${NC}"
    echo -e "${YELLOW}  cd ${BUILD_DIR}${NC}"
    echo -e "${YELLOW}  make -j\$(nproc)${NC}"
    echo -e "${YELLOW}  make test${NC}"
    echo

    docker run -it --rm \
        --platform linux/amd64 \
        -u serac \
        -v "${SMITH_REPO_PATH}/smith:/home/serac/serac" \
        "${DOCKER_IMAGE}" \
        /bin/bash
else
    # Non-interactive mode - run build automatically
    echo "Non-interactive mode: Running build commands automatically"
    echo

    docker run --rm \
        --platform linux/amd64 \
        -u serac \
        -v "${SMITH_REPO_PATH}/smith:/home/serac/serac" \
        "${DOCKER_IMAGE}" \
        /bin/bash -c "
            set -e
            echo '=== Configuring Smith build ==='
            cd /home/serac/serac
            python3 ./config-build.py -hc host-configs/docker/${HOST_CONFIG} -bp ${BUILD_DIR} -ip ${INSTALL_DIR}

            echo ''
            echo '=== Building Smith ==='
            cd ${BUILD_DIR}
            make -j\$(nproc)

            echo ''
            echo '=== Running tests ==='
            make test
        "
fi
