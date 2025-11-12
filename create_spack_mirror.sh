#!/bin/bash
# Create a local Spack mirror for offline installation
# Run this on a machine WITH internet access

set -e

MIRROR_DIR="${1:-./smith_mirror}"

echo "Creating Spack mirror in: ${MIRROR_DIR}"

cd smith

# Create mirror of all dependencies
python3 ./scripts/uberenv/uberenv.py \
    --create-mirror \
    --mirror="${MIRROR_DIR}" \
    --spack-env-file=./scripts/spack/configs/docker/ubuntu24/spack.yaml \
    --project-json=.uberenv_config.json \
    --spec="~devtools~enzyme %gcc_13"

echo ""
echo "Mirror created! Transfer ${MIRROR_DIR} to your target machine."
echo ""
echo "Then on the target machine, run:"
echo "  ./build_smith.sh --mirror=${MIRROR_DIR}"
