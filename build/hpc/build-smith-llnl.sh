#!/bin/bash
# ==============================================================================
# Smith Container Build Script for LLNL HPC Systems
# ==============================================================================
#
# DESCRIPTION:
#   Builds Smith on LLNL HPC systems (LC clusters) using Singularity/Apptainer
#   containers or native builds with system modules.
#
# SUPPORTED SYSTEMS:
#   - quartz, ruby, jade (x86_64 clusters)
#   - lassen (IBM POWER9 + NVIDIA V100)
#   - rzadams, rzvernal (Intel Sapphire Rapids)
#
# USAGE:
#   ./build-smith-llnl.sh [OPTIONS]
#
# OPTIONS:
#   -m, --method METHOD   Build method: container, native (default: container)
#   -s, --system SYSTEM   HPC system: quartz, lassen, ruby, etc.
#   -c, --compiler COMP   Compiler: gcc, clang, cuda (default: gcc)
#   -j, --jobs N          Number of parallel build jobs (default: 16)
#   -b, --batch           Submit as batch job instead of interactive
#   -t, --test            Run tests after building
#   -h, --help            Show this help message
#
# EXAMPLES:
#   # Container build on quartz (interactive)
#   ./build-smith-llnl.sh --system quartz
#
#   # Native build with system GCC
#   ./build-smith-llnl.sh --method native --compiler gcc
#
#   # Batch job build on lassen with CUDA
#   ./build-smith-llnl.sh --system lassen --compiler cuda --batch
#
# CONTAINER IMAGES:
#   Container images must be converted from Docker to Singularity format.
#   Images are stored in: /usr/workspace/$USER/containers/
#
#   To convert a Docker image:
#     singularity build smith-clang19.sif docker://seracllnl/tpls:clang-19_10-09-25_23h-54m
#
# NATIVE BUILD:
#   Uses LC modules and spack to build dependencies.
#   Requires network access for initial spack setup.
#
# OUTPUT:
#   Build artifacts:   ~/fea-converge/smith-build/
#   Install location:  ~/fea-converge/smith-install/
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
BUILD_METHOD="container"
HPC_SYSTEM=""
COMPILER="gcc"
BUILD_JOBS=16
BATCH_MODE=false
RUN_TESTS=false

# Help message
show_help() {
    grep '^#' "$0" | grep -v '#!/bin/bash' | sed 's/^# //g; s/^#//g'
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -m|--method)
            BUILD_METHOD="$2"
            shift 2
            ;;
        -s|--system)
            HPC_SYSTEM="$2"
            shift 2
            ;;
        -c|--compiler)
            COMPILER="$2"
            shift 2
            ;;
        -j|--jobs)
            BUILD_JOBS="$2"
            shift 2
            ;;
        -b|--batch)
            BATCH_MODE=true
            shift
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

# Auto-detect system if not specified
if [[ -z "$HPC_SYSTEM" ]]; then
    if [[ -f /etc/cluster/name ]]; then
        HPC_SYSTEM=$(cat /etc/cluster/name)
        echo -e "${YELLOW}Auto-detected system: ${HPC_SYSTEM}${NC}"
    elif [[ -n "$SYS_TYPE" ]]; then
        HPC_SYSTEM="$SYS_TYPE"
        echo -e "${YELLOW}Detected from SYS_TYPE: ${HPC_SYSTEM}${NC}"
    else
        echo -e "${RED}Error: Could not detect HPC system${NC}"
        echo "Please specify with --system"
        exit 1
    fi
fi

# Paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SMITH_DIR="${REPO_ROOT}/smith"
BUILD_DIR="${REPO_ROOT}/smith-build"
INSTALL_DIR="${REPO_ROOT}/smith-install"

# Print configuration
echo -e "${BLUE}=== Smith HPC Build for LLNL Systems ===${NC}"
echo
echo "Configuration:"
echo "  HPC System:       ${HPC_SYSTEM}"
echo "  Build Method:     ${BUILD_METHOD}"
echo "  Compiler:         ${COMPILER}"
echo "  Smith Source:     ${SMITH_DIR}"
echo "  Build Directory:  ${BUILD_DIR}"
echo "  Install Directory: ${INSTALL_DIR}"
echo "  Build Jobs:       ${BUILD_JOBS}"
echo "  Batch Mode:       ${BATCH_MODE}"
echo "  Run Tests:        ${RUN_TESTS}"
echo

# Check if Smith submodule exists
if [[ ! -f "${SMITH_DIR}/CMakeLists.txt" ]]; then
    echo -e "${RED}Error: Smith submodule not found.${NC}"
    echo "Please run: git submodule update --init --recursive"
    exit 1
fi

# Create build directories
mkdir -p "${BUILD_DIR}"
mkdir -p "${INSTALL_DIR}"

# Container build
if [[ "$BUILD_METHOD" == "container" ]]; then
    CONTAINER_DIR="${HOME}/containers"
    mkdir -p "${CONTAINER_DIR}"

    # Determine container image
    case "$COMPILER" in
        gcc)
            CONTAINER_IMAGE="${CONTAINER_DIR}/smith-gcc14.sif"
            HOST_CONFIG="gcc@14.2.0.cmake"
            ;;
        clang)
            CONTAINER_IMAGE="${CONTAINER_DIR}/smith-clang19.sif"
            HOST_CONFIG="llvm@19.1.1.cmake"
            ;;
        cuda)
            CONTAINER_IMAGE="${CONTAINER_DIR}/smith-cuda12.sif"
            HOST_CONFIG="gcc@12.3.0_cuda.cmake"
            ;;
        *)
            echo -e "${RED}Error: Unknown compiler '$COMPILER'${NC}"
            exit 1
            ;;
    esac

    # Check if container exists
    if [[ ! -f "${CONTAINER_IMAGE}" ]]; then
        echo -e "${RED}Error: Container image not found: ${CONTAINER_IMAGE}${NC}"
        echo
        echo "To create the container image, run:"
        case "$COMPILER" in
            gcc)
                echo "  singularity build ${CONTAINER_IMAGE} docker://seracllnl/tpls:gcc-14_10-09-25_23h-54m"
                ;;
            clang)
                echo "  singularity build ${CONTAINER_IMAGE} docker://seracllnl/tpls:clang-19_10-09-25_23h-54m"
                ;;
            cuda)
                echo "  singularity build ${CONTAINER_IMAGE} docker://seracllnl/tpls:cuda-12_04-16-25_20h-55m"
                ;;
        esac
        exit 1
    fi

    echo -e "${GREEN}✓ Using container: ${CONTAINER_IMAGE}${NC}"
    echo

    # Build command for container
    BUILD_CMD="cd /home/serac/serac && \
        python3 ./config-build.py -hc host-configs/docker/${HOST_CONFIG} -bp /tmp/smith-build -ip /tmp/smith-install && \
        cd /tmp/smith-build && \
        make -j${BUILD_JOBS} && \
        make install"

    if [[ "$RUN_TESTS" == "true" ]]; then
        BUILD_CMD+=" && ctest --output-on-failure -j${BUILD_JOBS}"
    fi

    # Run with Singularity
    if [[ "$BATCH_MODE" == "true" ]]; then
        # Create batch script
        BATCH_SCRIPT="${BUILD_DIR}/build-smith.sbatch"
        cat > "${BATCH_SCRIPT}" << EOF
#!/bin/bash
#SBATCH --job-name=smith-build
#SBATCH --nodes=1
#SBATCH --time=02:00:00
#SBATCH --output=${BUILD_DIR}/build-%j.out

singularity exec \\
    --bind ${SMITH_DIR}:/home/serac/serac:ro \\
    --bind ${BUILD_DIR}:/tmp/smith-build \\
    --bind ${INSTALL_DIR}:/tmp/smith-install \\
    ${CONTAINER_IMAGE} \\
    /bin/bash -c "${BUILD_CMD}"
EOF

        echo "Submitting batch job..."
        sbatch "${BATCH_SCRIPT}"
        echo -e "${GREEN}✓ Job submitted. Monitor with: squeue -u \$USER${NC}"
    else
        # Interactive build
        singularity exec \
            --bind ${SMITH_DIR}:/home/serac/serac:ro \
            --bind ${BUILD_DIR}:/tmp/smith-build \
            --bind ${INSTALL_DIR}:/tmp/smith-install \
            ${CONTAINER_IMAGE} \
            /bin/bash -c "${BUILD_CMD}"
    fi

# Native build
elif [[ "$BUILD_METHOD" == "native" ]]; then
    echo -e "${YELLOW}Native build not yet implemented for LLNL systems${NC}"
    echo "This will require:"
    echo "  1. Loading appropriate modules (gcc, cmake, python, mpi)"
    echo "  2. Running uberenv to build TPLs"
    echo "  3. Configuring and building Smith"
    echo
    echo "For now, please use container method: --method container"
    exit 1
else
    echo -e "${RED}Error: Unknown build method '$BUILD_METHOD'${NC}"
    echo "Available methods: container, native"
    exit 1
fi

# Build status
if [[ $? -eq 0 ]]; then
    echo
    echo -e "${GREEN}✓ Smith build complete!${NC}"
    echo
    echo "Build artifacts:"
    echo "  Build:   ${BUILD_DIR}"
    echo "  Install: ${INSTALL_DIR}"
    echo
else
    echo
    echo -e "${RED}✗ Smith build failed${NC}"
    exit 1
fi
