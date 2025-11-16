# Claude Code Context

## Project Overview

This repository provides **four distinct Bayesian optimization methods** for resolving contact convergence failures in finite element simulations using the LLNL Tribol contact library and Smith/Serac solver framework.

The Smith SDK is pulled in as a git submodule in ./smith.

The ./methods directory includes four independent ML methods for resolving contact convergence issues.

The ./smith-models directory includes contact exemplar problems defined using the Smith SDK.

The ./references directory contains PDFs that were used to construct the Smith models.

DO NOT WRITE MARKDOWN FILES UNLESS EXPLICITLY PROMPTED!

# Building Smith

Additional context for building Smith with containers is located here:

- https://serac.readthedocs.io/en/latest/sphinx/quickstart.html#using-a-docker-image-with-preinstalled-dependencies

Additional context for building Smith on LC HPC machines is here:

- https://serac.readthedocs.io/en/latest/sphinx/quickstart.html#building-smith

Additional context for using docker/podman on the LC is here (see the Special Considerations section):

- https://hpc.llnl.gov/documentation/user-guides/using-containers-lc-hpc-systems/containers-how-build-container
