#!/bin/bash
# Run Smith contact model inside Docker

set -e

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

if [ $# -eq 0 ]; then
    echo -e "${RED}Error: No model specified${NC}"
    echo "Usage: $0 <model_name>"
    echo ""
    echo "Available models:"
    echo "  die-on-slab"
    echo "  block-on-slab"
    echo "  sphere-in-sphere"
    exit 1
fi

MODEL_NAME=$1
REPO_PATH="$(cd "$(dirname "$0")" && pwd)"

echo -e "${GREEN}Running Smith model ${MODEL_NAME} in Docker...${NC}"

# Run the model build and execution in Docker
docker run --rm \
    --platform linux/amd64 \
    -v "${REPO_PATH}/smith:/home/serac/serac:ro" \
    -v "${REPO_PATH}/models:/home/serac/models:ro" \
    -v "${REPO_PATH}/smith-build:/home/serac/smith-build:ro" \
    -v "${REPO_PATH}/smith-install:/home/serac/smith-install:ro" \
    -v "${REPO_PATH}/output:/home/serac/output" \
    -w /home/serac \
    seracllnl/tpls:clang-19_10-09-25_23h-54m \
    /bin/bash -c "
        set -e

        MODEL=${MODEL_NAME}
        BUILD_DIR=build_\${MODEL}

        echo '=== Configuring model ==='
        mkdir -p \${BUILD_DIR}
        cd \${BUILD_DIR}

        cmake ../models/\${MODEL} \
            -C /home/serac/serac/host-configs/docker/llvm@19.1.1.cmake \
            -DSmith_DIR=/home/serac/smith-install/lib/cmake \
            -DCMAKE_BUILD_TYPE=Release

        echo ''
        echo '=== Building model ==='
        make -j\$(nproc)

        echo ''
        echo '=== Running model ==='
        ./\${MODEL//-/_}
    "

echo -e "${GREEN}✓ Model run complete${NC}"
