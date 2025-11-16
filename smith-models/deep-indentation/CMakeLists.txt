cmake_minimum_required(VERSION 3.14)

project(deep_indentation LANGUAGES C CXX)

# Find Smith
find_package(Smith REQUIRED)

# Create executable
add_executable(deep_indentation deep_indentation.cpp)

# Link Smith
target_link_libraries(deep_indentation PRIVATE Smith::Smith)

# Set C++ standard
set_target_properties(deep_indentation PROPERTIES
    CXX_STANDARD 17
    CXX_STANDARD_REQUIRED ON
)
