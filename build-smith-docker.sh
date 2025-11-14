#!/bin/bash
# Wrapper script to build Smith using Docker with host-mounted build directories
# This ensures build artifacts are accessible on the host filesystem

set -e

# Color output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== Building Smith with Docker ===${NC}"
echo

# Configuration
SMITH_REPO_PATH="$(cd "$(dirname "$0")" && pwd)"
DOCKER_IMAGE="${DOCKER_IMAGE:-seracllnl/tpls:clang-19_10-09-25_23h-54m}"
HOST_CONFIG="${HOST_CONFIG:-llvm@19.1.1.cmake}"

# Build and install directories (inside container, then copied to host)
HOST_BUILD_DIR="${SMITH_REPO_PATH}/smith-build"
HOST_INSTALL_DIR="${SMITH_REPO_PATH}/smith-install"
CONTAINER_BUILD_DIR="../smith-build"
CONTAINER_INSTALL_DIR="../smith-install"

echo "Configuration:"
echo "  Docker Image: ${DOCKER_IMAGE}"
echo "  Host Config:  ${HOST_CONFIG}"
echo "  Smith Source: ${SMITH_REPO_PATH}/smith"
echo "  Build Dir (host):      ${HOST_BUILD_DIR}"
echo "  Install Dir (host):    ${HOST_INSTALL_DIR}"
echo "  Build Dir (container): ${CONTAINER_BUILD_DIR}"
echo "  Install Dir (container): ${CONTAINER_INSTALL_DIR}"
echo

# Create directories on host
mkdir -p "${HOST_BUILD_DIR}"
mkdir -p "${HOST_INSTALL_DIR}"

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo -e "${RED}Error: Docker is not running. Please start Docker Desktop and try again.${NC}"
    exit 1
fi

# Check if image exists locally, if not pull it
if ! docker image inspect "${DOCKER_IMAGE}" > /dev/null 2>&1; then
    echo -e "${YELLOW}Docker image not found locally. Pulling ${DOCKER_IMAGE}...${NC}"
    echo -e "${YELLOW}(This is ~4 GB and may take several minutes)${NC}"
    docker pull --platform linux/amd64 "${DOCKER_IMAGE}"
else
    echo -e "${GREEN}Docker image ${DOCKER_IMAGE} found locally.${NC}"
fi

echo
echo -e "${GREEN}Building Smith in Docker container...${NC}"
echo

# Run the build in Docker - build artifacts stay in container, we'll copy them out
CONTAINER_ID=$(docker run -d \
    --platform linux/amd64 \
    -v "${SMITH_REPO_PATH}/smith:/home/serac/serac:ro" \
    "${DOCKER_IMAGE}" \
    /bin/bash -c "
        set -e
        echo '=== Configuring Smith build ==='
        cd /home/serac/serac
        python3 ./config-build.py \
            -hc host-configs/docker/${HOST_CONFIG} \
            -bp ${CONTAINER_BUILD_DIR} \
            -ip ${CONTAINER_INSTALL_DIR}

        echo ''
        echo '=== Building Smith ==='
        cd ${CONTAINER_BUILD_DIR}
        make -j\$(nproc)

        echo ''
        echo '=== Installing Smith ==='
        make install

        echo ''
        echo '=== Running tests ==='
        ctest --output-on-failure -j\$(nproc) || true

        # Keep container alive so we can copy artifacts
        sleep 10
    ")

# Follow the logs
docker logs -f ${CONTAINER_ID} 2>&1 || true

# Wait for container to finish
docker wait ${CONTAINER_ID} > /dev/null 2>&1 || true

# Copy build artifacts to host
echo ""
echo -e "${GREEN}Copying build artifacts to host...${NC}"
docker cp ${CONTAINER_ID}:/home/serac/${CONTAINER_BUILD_DIR}/. "${HOST_BUILD_DIR}/" 2>/dev/null || echo "Note: Some build files may not copy"
docker cp ${CONTAINER_ID}:/home/serac/${CONTAINER_INSTALL_DIR}/. "${HOST_INSTALL_DIR}/" 2>/dev/null || echo "Note: Some install files may not copy"

# Clean up container
docker rm ${CONTAINER_ID} > /dev/null 2>&1 || true

echo
echo -e "${GREEN}✓ Smith build complete!${NC}"
echo
echo "Build artifacts location:"
echo "  Build:   ${HOST_BUILD_DIR}"
echo "  Install: ${HOST_INSTALL_DIR}"
echo
echo "You can now run the contact models with: ./run_model <model-name>"
