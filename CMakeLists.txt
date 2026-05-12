cmake_minimum_required(VERSION 3.24)

project(xushi2
    VERSION 0.0.1
    LANGUAGES CXX
    DESCRIPTION "Xushi2 — deterministic 2D 3v3 hero-shooter MARL environment"
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Determinism flags — see docs/determinism_rules.md and docs/rl_design.md §9.
# NEVER pass -ffast-math. Precise floats are a project-level invariant.
if(MSVC)
    add_compile_options(/fp:precise /fp:except-)
    add_compile_options(/W4)
else()
    add_compile_options(-fno-fast-math -fno-associative-math -fno-reciprocal-math)
    add_compile_options(-fsignaling-nans -fno-finite-math-only)
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# --- Options ---
option(XUSHI2_BUILD_VIEWER        "Build the raylib viewer"            ON)
option(XUSHI2_BUILD_PYTHON_MODULE "Build the pybind11 Python module"   ON)
option(XUSHI2_BUILD_TESTS         "Build unit / integration tests"     ON)
option(XUSHI2_BUILD_BENCHMARKS    "Build C++ benchmarks"               OFF)
option(XUSHI2_WARNINGS_AS_ERRORS  "Treat compiler warnings as errors"  OFF)

# Static libs (xushi2_sim, xushi2_bots) are linked into the pybind11 shared
# module, so they must be position-independent on ELF platforms.
if(XUSHI2_BUILD_PYTHON_MODULE)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
endif()

if(XUSHI2_WARNINGS_AS_ERRORS)
    if(MSVC)
        add_compile_options(/WX)
    else()
        add_compile_options(-Werror)
    endif()
endif()

include(FetchContent)

set(FETCHCONTENT_TRY_FIND_PACKAGE_MODE ALWAYS)

# --- raylib (viewer only) ---
if(XUSHI2_BUILD_VIEWER)
    find_package(raylib 5.0 QUIET)
    if(NOT raylib_FOUND)
        set(RAYLIB_VERSION 5.0)
        FetchContent_Declare(
            raylib
            GIT_REPOSITORY https://github.com/raysan5/raylib.git
            GIT_TAG        ${RAYLIB_VERSION}
            GIT_SHALLOW    TRUE
            FIND_PACKAGE_ARGS 5.0
        )
        set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
        set(BUILD_GAMES    OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(raylib)
    endif()
endif()

# --- pybind11 (Python module only) ---
if(XUSHI2_BUILD_PYTHON_MODULE)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG        v2.12.0
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(pybind11)
endif()

# --- GoogleTest (tests only) ---

# --- Google Benchmark (benchmarks only) ---
if(XUSHI2_BUILD_BENCHMARKS)
    find_package(benchmark QUIET)
    if(NOT benchmark_FOUND)
        FetchContent_Declare(
            googlebenchmark
            GIT_REPOSITORY https://github.com/google/benchmark.git
            GIT_TAG        v1.9.0
            GIT_SHALLOW    TRUE
            FIND_PACKAGE_ARGS NAMES benchmark
        )
        set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
        set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googlebenchmark)
    endif()
endif()

if(XUSHI2_BUILD_TESTS)
    find_package(GTest QUIET)
    if(NOT GTest_FOUND)
        FetchContent_Declare(
            googletest
            GIT_REPOSITORY https://github.com/google/googletest.git
            GIT_TAG        v1.14.0
            GIT_SHALLOW    TRUE
            FIND_PACKAGE_ARGS NAMES GTest
        )
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
    endif()
    enable_testing()
endif()

add_subdirectory(src)

if(XUSHI2_BUILD_TESTS)
    add_subdirectory(tests)
endif()

if(XUSHI2_BUILD_BENCHMARKS)
    add_subdirectory(benchmarks)
endif()
