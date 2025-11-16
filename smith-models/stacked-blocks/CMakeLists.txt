cmake_minimum_required(VERSION 3.14)

project(stacked_blocks LANGUAGES C CXX)

# Find Smith
find_package(Smith REQUIRED)

# Create executable
add_executable(stacked_blocks stacked_blocks.cpp)

# Link Smith
target_link_libraries(stacked_blocks PRIVATE Smith::Smith)

# Set C++ standard
set_target_properties(stacked_blocks PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
