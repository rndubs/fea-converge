#!/bin/bash
# Simple Smith build script with mounted directories
set -e

REPO_PATH="$(cd "$(dirname "$0")" && pwd)"

echo "Building Smith with mounted directories..."

docker run --rm \
    --platform linux/amd64 \
    -v "${REPO_PATH}/smith:/home/serac/serac:ro" \
    -v "${REPO_PATH}/smith-build:/home/serac/smith-build" \
    -v "${REPO_PATH}/smith-install:/home/serac/smith-install" \
    seracllnl/tpls:clang-19_10-09-25_23h-54m \
    /bin/bash -c '
        set -e
        cd /home/serac/serac
        python3 ./config-build.py \
            -hc host-configs/docker/llvm@19.1.1.cmake \
            -bp /home/serac/smith-build \
            -ip /home/serac/smith-install

        cd /home/serac/smith-build
        make -j$(nproc)
        make install
    '

echo "✓ Smith built and installed successfully!"
