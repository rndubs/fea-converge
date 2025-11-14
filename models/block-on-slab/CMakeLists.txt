cmake_minimum_required(VERSION 3.14)

project(block_on_slab LANGUAGES C CXX)

message(STATUS "CMake Version: ${CMAKE_VERSION}")

# Find Smith
include(CMakeFindDependencyMacro)
if(DEFINED Smith_DIR)
    set(SMITH_DIR ${Smith_DIR})
endif()
find_dependency(smith REQUIRED NO_DEFAULT_PATH PATHS "${SMITH_DIR}" "${SMITH_DIR}/lib/cmake")

# Remove non-existent INTERFACE_INCLUDE_DIRECTORIES from imported targets
# to work around CMake error
set(_imported_targets
   axom
   axom::mfem
   conduit
   conduit::conduit_mpi
   conduit::conduit
   conduit_relay_mpi
   conduit_relay_mpi_io
   conduit_blueprint
   conduit_blueprint_mpi
   tribol::mfem
   smith::mfem
   mfem)

foreach(_target ${_imported_targets})
   if(TARGET ${_target})
      message(STATUS "Removing non-existent include directories from target[${_target}]")

      get_target_property(_dirs ${_target} INTERFACE_INCLUDE_DIRECTORIES)
      set(_existing_dirs)
      foreach(_dir ${_dirs})
         if (EXISTS "${_dir}")
            list(APPEND _existing_dirs "${_dir}")
         endif()
      endforeach()
      if (_existing_dirs)
         set_target_properties(${_target} PROPERTIES
                               INTERFACE_INCLUDE_DIRECTORIES "${_existing_dirs}" )
      endif()
   endif()
endforeach()

# Add executable
add_executable(block_on_slab block_on_slab.cpp)

# Link against Smith
target_link_libraries(block_on_slab smith::smith)

# Set output directory
set_target_properties(block_on_slab PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}"
)
