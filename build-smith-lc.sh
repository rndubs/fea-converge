#!/bin/bash
# Copyright (c) Lawrence Livermore National Security, LLC
# SPDX-License-Identifier: (BSD-3-Clause)
#
# Build Smith on Lawrence Livermore Computing (LC) HPC systems

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Detect LC machine
MACHINE=$(hostname | sed 's/[0-9]*$//')

# Default configuration
BUILD_DIR="${BUILD_DIR:-$(pwd)/smith-build}"
INSTALL_DIR="${INSTALL_DIR:-$(pwd)/smith-install}"
SMITH_SOURCE="$(pwd)/smith"

# Machine-specific defaults
case "$MACHINE" in
    dane|rzgenie|rzwhippet)
        DEFAULT_COMPILER="llvm@19.1.3"
        SCHEDULER="slurm"
        ;;
    rzvernal|tioga)
        DEFAULT_COMPILER="rocmcc@6.2.1_hip"
        SCHEDULER="slurm"
        ;;
    rzansel|lassen)
        DEFAULT_COMPILER="clang@14.0.5_cuda"
        SCHEDULER="lsf"
        ;;
    *)
        echo -e "${YELLOW}Warning: Unknown LC machine '$MACHINE'${NC}"
        echo "Defaulting to llvm compiler"
        DEFAULT_COMPILER="llvm@19.1.3"
        SCHEDULER="slurm"
        ;;
esac

# Allow override via environment
COMPILER="${COMPILER:-$DEFAULT_COMPILER}"

# Usage information
usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Build Smith on LC HPC systems

Options:
  --compiler COMPILER     Compiler to use (default: auto-detected for machine)
  --build-dir DIR         Build directory (default: ./smith-build)
  --install-dir DIR       Install directory (default: ./smith-install)
  --interactive           Run build interactively on current node
  --batch                 Submit as batch job (default)
  --clean                 Clean build directories before building
  -h, --help             Show this help message

Detected Configuration:
  Machine:      $MACHINE
  Scheduler:    $SCHEDULER
  Compiler:     $COMPILER
  Build Dir:    $BUILD_DIR
  Install Dir:  $INSTALL_DIR

Available Compilers:
  - llvm@19.1.3 (dane, rzwhippet, rzgenie)
  - gcc@13.3.1 (dane, rzwhippet)
  - rocmcc@6.2.1_hip (rzvernal, tioga - AMD GPU)
  - clang@14.0.5_cuda (rzansel, lassen - NVIDIA GPU)

Examples:
  $0                              # Auto-detect and submit batch job
  $0 --interactive                # Build interactively on current node
  $0 --compiler gcc@13.3.1        # Use specific compiler
  $0 --clean                      # Clean and rebuild

EOF
}

# Parse arguments
MODE="batch"
CLEAN=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --compiler)
            COMPILER="$2"
            shift 2
            ;;
        --build-dir)
            BUILD_DIR="$2"
            shift 2
            ;;
        --install-dir)
            INSTALL_DIR="$2"
            shift 2
            ;;
        --interactive)
            MODE="interactive"
            shift
            ;;
        --batch)
            MODE="batch"
            shift
            ;;
        --clean)
            CLEAN=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo -e "${RED}Error: Unknown option $1${NC}"
            usage
            exit 1
            ;;
    esac
done

# Clean if requested
if [ "$CLEAN" = true ]; then
    echo -e "${YELLOW}Cleaning build directories...${NC}"
    rm -rf "$BUILD_DIR" "$INSTALL_DIR"
    echo -e "${GREEN}✓ Clean complete${NC}"
fi

# Determine host config file
if [ -f "$SMITH_SOURCE/host-configs/${MACHINE}-*.cmake" ]; then
    # Find matching host config
    HOST_CONFIG=$(ls "$SMITH_SOURCE/host-configs/${MACHINE}"*"${COMPILER}"*.cmake 2>/dev/null | head -1)
    if [ -z "$HOST_CONFIG" ]; then
        # Try generic machine name
        HOST_CONFIG=$(ls "$SMITH_SOURCE/host-configs/${MACHINE}"*.cmake 2>/dev/null | head -1)
    fi
else
    echo -e "${RED}Error: No host config found for machine '$MACHINE'${NC}"
    echo "Available configs:"
    ls "$SMITH_SOURCE/host-configs/"*.cmake 2>/dev/null | xargs -n 1 basename
    exit 1
fi

if [ -z "$HOST_CONFIG" ]; then
    echo -e "${RED}Error: Could not find host config for $MACHINE with $COMPILER${NC}"
    exit 1
fi

HOST_CONFIG_NAME=$(basename "$HOST_CONFIG")

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}Building Smith on LC${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Machine:      $MACHINE"
echo "Compiler:     $COMPILER"
echo "Host Config:  $HOST_CONFIG_NAME"
echo "Build Dir:    $BUILD_DIR"
echo "Install Dir:  $INSTALL_DIR"
echo "Mode:         $MODE"
echo ""

# Build function
do_build() {
    echo -e "${BLUE}==> Configuring Smith with CMake...${NC}"

    cd "$SMITH_SOURCE"
    python3 ./config-build.py \
        -hc "$HOST_CONFIG" \
        -bp "$BUILD_DIR" \
        -ip "$INSTALL_DIR" \
        -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

    echo ""
    echo -e "${BLUE}==> Building Smith...${NC}"
    cd "$BUILD_DIR"

    # Use parallel build based on available cores
    if command -v nproc &> /dev/null; then
        NCORES=$(nproc)
    else
        NCORES=8
    fi

    make -j${NCORES}

    echo ""
    echo -e "${BLUE}==> Installing Smith...${NC}"
    make install

    echo ""
    echo -e "${GREEN}========================================${NC}"
    echo -e "${GREEN}Smith built and installed successfully!${NC}"
    echo -e "${GREEN}========================================${NC}"
    echo ""
    echo "Installation directory: $INSTALL_DIR"
    echo ""
    echo "You can now run models with:"
    echo "  ./run_model_lc <model-name>"
    echo ""
}

# Interactive build
if [ "$MODE" = "interactive" ]; then
    echo -e "${YELLOW}Starting interactive build...${NC}"
    echo -e "${YELLOW}This will take approximately 30-60 minutes...${NC}"
    echo ""
    do_build

elif [ "$MODE" = "batch" ]; then
    # Create batch script
    BATCH_SCRIPT="build_smith_${MACHINE}_$$.sh"

    if [ "$SCHEDULER" = "slurm" ]; then
        # SLURM batch script
        cat > "$BATCH_SCRIPT" << 'EOFBATCH'
#!/bin/bash
#SBATCH --job-name=build-smith
#SBATCH --output=build-smith-%j.out
#SBATCH --error=build-smith-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=36
#SBATCH --time=02:00:00
#SBATCH --partition=pdebug

set -e

echo "=================================================="
echo "Building Smith on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "=================================================="
echo ""

EOFBATCH
    elif [ "$SCHEDULER" = "lsf" ]; then
        # LSF batch script
        cat > "$BATCH_SCRIPT" << 'EOFBATCH'
#!/bin/bash
#BSUB -J build-smith
#BSUB -o build-smith-%J.out
#BSUB -e build-smith-%J.err
#BSUB -nnodes 1
#BSUB -W 120
#BSUB -q pdebug

set -e

echo "=================================================="
echo "Building Smith on $(hostname)"
echo "Job ID: $LSB_JOBID"
echo "=================================================="
echo ""

EOFBATCH
    fi

    # Add build commands to batch script
    cat >> "$BATCH_SCRIPT" << EOFBATCH

# Navigate to project directory
cd $(pwd)

# Export variables
export BUILD_DIR="$BUILD_DIR"
export INSTALL_DIR="$INSTALL_DIR"
export SMITH_SOURCE="$SMITH_SOURCE"
export HOST_CONFIG="$HOST_CONFIG"

# Build function
$(declare -f do_build)

# Execute build
do_build

echo ""
echo "Build completed at: \$(date)"
EOFBATCH

    chmod +x "$BATCH_SCRIPT"

    echo -e "${BLUE}Batch script created: $BATCH_SCRIPT${NC}"
    echo ""

    # Submit job
    if [ "$SCHEDULER" = "slurm" ]; then
        echo -e "${YELLOW}Submitting batch job via SLURM...${NC}"
        JOB_ID=$(sbatch "$BATCH_SCRIPT" | awk '{print $NF}')
        echo -e "${GREEN}Job submitted with ID: $JOB_ID${NC}"
        echo ""
        echo "Monitor job with:"
        echo "  squeue -j $JOB_ID"
        echo "  tail -f build-smith-${JOB_ID}.out"
    elif [ "$SCHEDULER" = "lsf" ]; then
        echo -e "${YELLOW}Submitting batch job via LSF...${NC}"
        JOB_ID=$(bsub < "$BATCH_SCRIPT" | grep -oP '(?<=Job <)\d+')
        echo -e "${GREEN}Job submitted with ID: $JOB_ID${NC}"
        echo ""
        echo "Monitor job with:"
        echo "  bjobs $JOB_ID"
        echo "  tail -f build-smith-${JOB_ID}.out"
    fi

    echo ""
    echo "Batch script will be automatically removed after job completes."
fi
