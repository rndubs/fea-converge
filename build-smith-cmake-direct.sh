#!/bin/bash
# Build Smith using cmake directly, bypassing config-build.py
set -e

REPO_PATH="$(cd "$(dirname "$0")" && pwd)"

echo "Building Smith with cmake directly..."

docker run --rm \
    --platform linux/amd64 \
    -v "${REPO_PATH}/smith:/home/serac/serac:ro" \
    -v "${REPO_PATH}/smith-build:/home/serac/smith-build" \
    -v "${REPO_PATH}/smith-install:/home/serac/smith-install" \
    seracllnl/tpls:clang-19_10-09-25_23h-54m \
    /bin/bash -c '
        set -e
        cd /home/serac/smith-build

        # Run cmake directly with the host config
        cmake \
            -C /home/serac/serac/host-configs/docker/llvm@19.1.1.cmake \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_INSTALL_PREFIX=/home/serac/smith-install \
            /home/serac/serac

        make -j$(nproc)
        make install
    '

echo "✓ Smith built and installed successfully!"
