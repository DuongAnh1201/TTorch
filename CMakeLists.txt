cmake_minimum_required(VERSION 3.18)

project(ttorch LANGUAGES CXX)

# Require modern C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Suppress CMP0148 deprecation warning from pybind11's FindPythonLibsNew
cmake_policy(SET CMP0148 OLD)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find dependencies
find_package(pybind11 CONFIG REQUIRED)

# Source files
set(SOURCES
    src/tensor.cpp
    src/autograd.cpp
    src/bindings.cpp
)

# Build Python module
pybind11_add_module(ttorch ${SOURCES})

# Include header directory
target_include_directories(ttorch PRIVATE include)

# Optional optimization flags
target_compile_options(ttorch PRIVATE
    $<$<CONFIG:Release>:-O3>
)
