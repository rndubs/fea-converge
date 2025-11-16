cmake_minimum_required(VERSION 3.14)

project(hemisphere_twisting LANGUAGES C CXX)

# Find Smith
find_package(Smith REQUIRED)

# Create executable
add_executable(hemisphere_twisting hemisphere_twisting.cpp)

# Link Smith
target_link_libraries(hemisphere_twisting PRIVATE Smith::Smith)

# Set C++ standard
set_target_properties(hemisphere_twisting PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
