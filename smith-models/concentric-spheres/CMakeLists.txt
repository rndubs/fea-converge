cmake_minimum_required(VERSION 3.14)

project(concentric_spheres LANGUAGES CXX)

# Find Smith
find_package(Smith REQUIRED)

# Create executable
add_executable(concentric_spheres concentric_spheres.cpp)

# Link Smith
target_link_libraries(concentric_spheres PRIVATE Smith::Smith)

# Set C++ standard
set_target_properties(concentric_spheres PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
