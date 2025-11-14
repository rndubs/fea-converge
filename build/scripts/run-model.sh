#!/bin/bash
# ==============================================================================
# Smith Model Run Script
# ==============================================================================
#
# DESCRIPTION:
#   Runs a compiled contact model from the smith-models directory.
#   Works on both macOS (Docker) and LLNL HPC (Singularity/native).
#
# USAGE:
#   ./run-model.sh <model-name> [OPTIONS]
#
# ARGUMENTS:
#   model-name          Name of the model to run (required)
#
# OPTIONS:
#   -e, --env ENV       Run environment: docker, singularity, native (auto-detect)
#   -n, --np N          Number of MPI ranks (default: 1)
#   -o, --output DIR    Output directory (default: model build directory)
#   -p, --params FILE   Parameter file for model (optional)
#   -b, --batch         Submit as batch job on HPC (LLNL only)
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
#   # Run die-on-slab model
#   ./run-model.sh die-on-slab
#
#   # Run with 4 MPI ranks
#   ./run-model.sh sphere-in-sphere --np 4
#
#   # Run as batch job on HPC
#   ./run-model.sh block-on-slab --batch
#
# PREREQUISITES:
#   Model must be built first:
#     ./build-model.sh <model-name>
#
# OUTPUT:
#   Results:  <model-build-dir>/<model_name>_paraview.*
#   Logs:     <model-build-dir>/<model_name>.log
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
RUN_ENV="auto"
MPI_RANKS=1
OUTPUT_DIR=""
PARAM_FILE=""
BATCH_MODE=false

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
            RUN_ENV="$2"
            shift 2
            ;;
        -n|--np)
            MPI_RANKS="$2"
            shift 2
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -p|--params)
            PARAM_FILE="$2"
            shift 2
            ;;
        -b|--batch)
            BATCH_MODE=true
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
MODEL_BUILD_DIR="${MODELS_DIR}/build/${MODEL_NAME}"
SMITH_INSTALL_DIR="${REPO_ROOT}/smith-install"

# Set output directory
if [[ -z "${OUTPUT_DIR}" ]]; then
    OUTPUT_DIR="${MODEL_BUILD_DIR}"
fi

# Validate model is built
if [[ ! -d "${MODEL_BUILD_DIR}" ]]; then
    echo -e "${RED}Error: Model '${MODEL_NAME}' not built${NC}"
    echo "Expected build directory: ${MODEL_BUILD_DIR}"
    echo
    echo "Please build the model first:"
    echo "  build/scripts/build-model.sh ${MODEL_NAME}"
    exit 1
fi

# Find the executable
MODEL_EXECUTABLE=$(find "${MODEL_BUILD_DIR}" -maxdepth 1 -type f -executable 2>/dev/null | head -1)
if [[ -z "${MODEL_EXECUTABLE}" || ! -x "${MODEL_EXECUTABLE}" ]]; then
    echo -e "${RED}Error: Model executable not found in ${MODEL_BUILD_DIR}${NC}"
    echo "Please rebuild the model:"
    echo "  build/scripts/build-model.sh ${MODEL_NAME}"
    exit 1
fi

# Auto-detect run environment
if [[ "$RUN_ENV" == "auto" ]]; then
    if command -v docker &> /dev/null && docker info &> /dev/null 2>&1; then
        RUN_ENV="docker"
    elif command -v singularity &> /dev/null; then
        RUN_ENV="singularity"
    else
        RUN_ENV="native"
    fi
    echo -e "${YELLOW}Auto-detected run environment: ${RUN_ENV}${NC}"
fi

# Print configuration
echo -e "${BLUE}=== Running Smith Model ===${NC}"
echo
echo "Configuration:"
echo "  Model:            ${MODEL_NAME}"
echo "  Build Directory:  ${MODEL_BUILD_DIR}"
echo "  Executable:       ${MODEL_EXECUTABLE}"
echo "  Output Directory: ${OUTPUT_DIR}"
echo "  Run Environment:  ${RUN_ENV}"
echo "  MPI Ranks:        ${MPI_RANKS}"
if [[ -n "${PARAM_FILE}" ]]; then
    echo "  Parameter File:   ${PARAM_FILE}"
fi
echo "  Batch Mode:       ${BATCH_MODE}"
echo

# Prepare run command
RUN_CMD=""
if [[ ${MPI_RANKS} -gt 1 ]]; then
    RUN_CMD="mpirun -np ${MPI_RANKS} "
fi

EXECUTABLE_NAME=$(basename "${MODEL_EXECUTABLE}")
RUN_CMD+="${EXECUTABLE_NAME}"

if [[ -n "${PARAM_FILE}" ]]; then
    RUN_CMD+=" --params ${PARAM_FILE}"
fi

# Run based on environment
case "$RUN_ENV" in
    docker)
        # Docker run (macOS)
        DOCKER_IMAGE="seracllnl/tpls:clang-19_10-09-25_23h-54m"

        if ! docker image inspect "${DOCKER_IMAGE}" > /dev/null 2>&1; then
            echo -e "${RED}Error: Docker image not found: ${DOCKER_IMAGE}${NC}"
            exit 1
        fi

        echo -e "${GREEN}Running model in Docker...${NC}"
        echo

        DOCKER_BINDS="-v ${MODEL_BUILD_DIR}:/workspace:rw"
        if [[ -n "${PARAM_FILE}" && -f "${PARAM_FILE}" ]]; then
            DOCKER_BINDS+=" -v ${PARAM_FILE}:/params:ro"
        fi

        docker run --rm \
            --platform linux/amd64 \
            -u serac \
            ${DOCKER_BINDS} \
            -w /workspace \
            "${DOCKER_IMAGE}" \
            /bin/bash -c "${RUN_CMD}"
        ;;

    singularity)
        # Singularity run (LLNL HPC)
        CONTAINER_IMAGE="${HOME}/containers/smith-clang19.sif"

        if [[ ! -f "${CONTAINER_IMAGE}" ]]; then
            echo -e "${RED}Error: Singularity container not found: ${CONTAINER_IMAGE}${NC}"
            exit 1
        fi

        if [[ "$BATCH_MODE" == "true" ]]; then
            # Create batch script
            BATCH_SCRIPT="${MODEL_BUILD_DIR}/run-${MODEL_NAME}.sbatch"
            cat > "${BATCH_SCRIPT}" << EOF
#!/bin/bash
#SBATCH --job-name=${MODEL_NAME}
#SBATCH --nodes=1
#SBATCH --ntasks=${MPI_RANKS}
#SBATCH --time=01:00:00
#SBATCH --output=${OUTPUT_DIR}/${MODEL_NAME}-%j.out

module load singularity

cd ${MODEL_BUILD_DIR}
singularity exec \\
    --bind ${MODEL_BUILD_DIR}:/workspace \\
    ${CONTAINER_IMAGE} \\
    /bin/bash -c "cd /workspace && ${RUN_CMD}"
EOF

            echo "Submitting batch job..."
            sbatch "${BATCH_SCRIPT}"
            echo -e "${GREEN}✓ Job submitted. Monitor with: squeue -u \$USER${NC}"
            exit 0
        else
            # Interactive run
            echo -e "${GREEN}Running model with Singularity...${NC}"
            echo

            cd "${MODEL_BUILD_DIR}"
            singularity exec \
                --bind ${MODEL_BUILD_DIR}:/workspace \
                ${CONTAINER_IMAGE} \
                /bin/bash -c "cd /workspace && ${RUN_CMD}"
        fi
        ;;

    native)
        # Native run
        echo -e "${GREEN}Running model natively...${NC}"
        echo

        cd "${MODEL_BUILD_DIR}"
        eval "${RUN_CMD}"
        ;;

    *)
        echo -e "${RED}Error: Unknown run environment '${RUN_ENV}'${NC}"
        echo "Available environments: docker, singularity, native"
        exit 1
        ;;
esac

# Run status
if [[ $? -eq 0 ]]; then
    echo
    echo -e "${GREEN}✓ Model run complete!${NC}"
    echo
    echo "Output files:"
    echo "  Directory: ${OUTPUT_DIR}"
    echo
    echo "Visualization:"
    echo "  ParaView files: ${OUTPUT_DIR}/${MODEL_NAME//-/_}_paraview.*"
    echo
    echo "To visualize:"
    echo "  1. Open ParaView"
    echo "  2. Load: ${OUTPUT_DIR}/${MODEL_NAME//-/_}_paraview.pvd"
    echo "  3. Click 'Apply' and use time controls to step through simulation"
else
    echo
    echo -e "${RED}✗ Model run failed${NC}"
    exit 1
fi
