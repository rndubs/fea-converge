#!/bin/bash
# Copyright (c) Lawrence Livermore National Security, LLC
# SPDX-License-Identifier: (BSD-3-Clause)
#
# Build Smith using Docker container with pre-installed TPLs

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default configuration
DOCKER_IMAGE="${DOCKER_IMAGE:-seracllnl/tpls:clang-19_latest}"
HOST_CONFIG="${HOST_CONFIG:-llvm@19.1.1.cmake}"
BUILD_DIR="$(pwd)/smith-build"
INSTALL_DIR="$(pwd)/smith-install"
SMITH_SOURCE="$(pwd)/smith"

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Building Smith with Docker${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Docker Image:    $DOCKER_IMAGE"
echo "Host Config:     $HOST_CONFIG"
echo "Build Dir:       $BUILD_DIR"
echo "Install Dir:     $INSTALL_DIR"
echo "Smith Source:    $SMITH_SOURCE"
echo ""

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}Error: Docker is not running${NC}"
    echo "Please start Docker Desktop and try again."
    exit 1
fi

# Check if Smith submodule is initialized
if [ ! -f "$SMITH_SOURCE/CMakeLists.txt" ]; then
    echo -e "${YELLOW}Smith submodule not initialized. Initializing...${NC}"
    git submodule update --init --recursive
fi

# Create build and install directories with appropriate permissions
echo -e "${GREEN}Creating build directories...${NC}"
mkdir -p "$BUILD_DIR"
mkdir -p "$INSTALL_DIR"
chmod -R 777 "$BUILD_DIR" "$INSTALL_DIR"

# Pull Docker image
echo -e "${GREEN}Pulling Docker image (this may take a while on first run)...${NC}"
docker pull "$DOCKER_IMAGE"

# Build Smith inside Docker container
echo -e "${GREEN}Building Smith inside Docker container...${NC}"
echo -e "${YELLOW}This will take approximately 15-30 minutes...${NC}"

docker run --rm \
    --platform linux/amd64 \
    -v "$(pwd):/home/serac/fea-converge" \
    -w /home/serac/fea-converge/smith \
    "$DOCKER_IMAGE" \
    bash -c "
        set -e
        echo '==> Configuring Smith with CMake...'
        python3 ./config-build.py \
            -hc host-configs/docker/$HOST_CONFIG \
            -bp /home/serac/fea-converge/smith-build \
            -ip /home/serac/fea-converge/smith-install \
            -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

        echo ''
        echo '==> Building Smith...'
        cd /home/serac/fea-converge/smith-build
        make -j\$(nproc)

        echo ''
        echo '==> Installing Smith...'
        make install

        echo ''
        echo '==> Build complete!'
    "

# Verify installation
if [ -f "$INSTALL_DIR/lib/cmake/smith-config.cmake" ]; then
    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Smith built and installed successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Installation directory: $INSTALL_DIR"
    echo ""
    echo "You can now run models with:"
    echo "  ./run_model <model-name>"
    echo ""
    echo "Or run all models with:"
    echo "  ./run_model --all"
    echo ""
else
    echo -e "${RED}Error: Smith installation failed${NC}"
    echo "Could not find smith-config.cmake in $INSTALL_DIR"
    exit 1
fi
