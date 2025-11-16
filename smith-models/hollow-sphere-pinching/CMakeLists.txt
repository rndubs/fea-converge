cmake_minimum_required(VERSION 3.14)

project(hollow_sphere_pinching LANGUAGES C CXX)

# Find Smith
find_package(Smith REQUIRED)

# Create executable
add_executable(hollow_sphere_pinching hollow_sphere_pinching.cpp)

# Link Smith
target_link_libraries(hollow_sphere_pinching PRIVATE Smith::Smith)

# Set C++ standard
set_target_properties(hollow_sphere_pinching PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
