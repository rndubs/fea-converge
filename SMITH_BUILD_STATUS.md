# Smith Build Status

## Summary

The Smith build system has been configured and prepared for building. All basic prerequisites are now installed and the submodules have been initialized.

## Fixed Issues

### 1. ✅ MPI Not Installed
**Problem:** The build script requires MPI (mpicc) but it was not installed.
**Solution:** Installed MPICH 4.2.0 with development headers:
```bash
apt-get install -y mpich libmpich-dev
```

### 2. ✅ Smith Submodule Not Initialized
**Problem:** The `smith` directory was empty because the git submodule wasn't initialized.
**Solution:** Initialized all submodules recursively:
```bash
git submodule update --init --recursive
```

### 3. ✅ Build Dependencies Verified
All required build tools are now present:
- **CMake:** 3.28.3 ✓
- **Python:** 3.11.14 ✓
- **GCC:** 13.3.0 ✓
- **gfortran:** 13.3.0 ✓
- **MPI:** MPICH 4.2.0 ✓
- **Clang:** 18.1.3 ✓

## Current Build Configuration

The `build_smith.sh` script uses the following configuration:
- **Spack Config:** `./scripts/spack/configs/docker/ubuntu24/spack.yaml`
- **Compiler:** gcc_13 (matches installed GCC 13.3.0)
- **Build Spec:** `~devtools~enzyme %gcc_13`
- **Build Jobs:** Auto-detected using `nproc`

## Remaining Limitations

### ⚠️ Network Access Required for Full Build

The Smith build process uses **uberenv/Spack** to download and build third-party libraries (TPLs). This requires:

1. **Internet access** to download Spack packages
2. **Access to external repositories** (GitHub, Spack mirrors)
3. **Ability to fetch source tarballs** for TPLs

**Important:** As noted in `CLAUDE.md`, the Smith build **cannot complete in Claude Code for the Web environments** due to network restrictions. The build will fail when uberenv attempts to:
- Clone Spack repositories
- Download package sources
- Fetch dependency metadata

### Alternative Build Approaches

For environments with network restrictions:

1. **Use Pre-built TPLs:**
   - Build TPLs on a machine with network access
   - Copy the `smith_tpls` directory to the restricted environment
   - Skip the uberenv step in the build script

2. **Spack Mirror:**
   - Create a local Spack mirror with all required packages
   - Use the `create_spack_mirror.sh` script (included in repo)
   - Configure Spack to use the local mirror

3. **Local Development Environment:**
   - Build Smith in a local Docker container with network access
   - Use the build for development/testing

## Build Process Steps

The `build_smith.sh` script performs these steps:

1. **Check Prerequisites:** Verify cmake, python3, MPI, gcc
2. **Build TPLs with uberenv:** Download and compile dependencies via Spack
3. **Configure Smith:** Generate CMake configuration using host-config file
4. **Build Smith:** Compile the Smith finite element solver
5. **Run Tests:** Execute the test suite to verify the build

## Spack Configuration Details

The build uses `docker/ubuntu24/spack.yaml` which:
- Defines compiler toolchains (gcc_13, gcc_14, clang_19)
- Specifies system packages that won't be rebuilt
- Configures MPICH as the MPI provider
- Sets build flags and architecture targets

Alternative config available: `linux_ubuntu_24/spack.yaml` for native Linux builds.

## Next Steps

To attempt the build (network access permitting):

```bash
./build_smith.sh
```

The script will:
- Detect and use all installed prerequisites
- Attempt to download dependencies via uberenv
- Build Smith if all dependencies are successfully obtained

**Expected Result in Restricted Environment:**
The build will fail during the uberenv step with network-related errors (connection timeouts, DNS failures, etc.).

## Verification Commands

Check that prerequisites are properly installed:

```bash
# Verify tools
cmake --version
python3 --version
mpicc --version
gcc --version
gfortran --version

# Verify submodule
git submodule status | grep smith

# Check uberenv exists
ls -la smith/scripts/uberenv/uberenv.py

# View config files
ls -la smith/scripts/spack/configs/docker/ubuntu24/
```

## Build Script Improvements

The `build_smith.sh` script includes:
- Automatic prerequisite checking
- Interactive prompts for rebuilding existing directories
- Verbose build output for debugging
- Parallel build support (using all available cores)
- Test execution after build completes

---

**Status:** Ready to attempt build (network limitations apply)
**Date:** 2025-11-12
