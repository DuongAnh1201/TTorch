cmake_minimum_required(VERSION 3.15)
project(ttorch LANGUAGES CXX)

# ---- C++ standard ----
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---- Find pybind11 ----
find_package(pybind11 REQUIRED)

# ---- Build the Python extension ----
pybind11_add_module(_C
    src/tensor.cpp
    src/autograd.cpp
    src/bindings.cpp
)

# ---- Include headers ----
target_include_directories(_C PRIVATE include)

# ---- Optimization ----
target_compile_options(_C PRIVATE
    $<$<CONFIG:Release>:-O3>
)

# ---- Install into Python package ----
# This is the CRITICAL part for scikit-build-core
install(TARGETS _C
    LIBRARY DESTINATION ttorch   # Linux/macOS (.so)
    ARCHIVE DESTINATION ttorch   # static libs (safe fallback)
    RUNTIME DESTINATION ttorch   # Windows (.pyd)
)