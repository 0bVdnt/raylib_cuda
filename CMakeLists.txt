cmake_minimum_required(VERSION 3.18)

project(raylib_cuda_lib
    VERSION 1.1.1
    DESCRIPTION "CUDA interop library for raylib"
    LANGUAGES C CXX CUDA
)

# ==================================================================
# Build Options
# ==================================================================

option(RLC_BUILD_TESTS    "Build test programs"    OFF)
option(RLC_BUILD_EXAMPLES "Build example programs" ON)

# ==================================================================
# CUDA Configuration
# ==================================================================

# Use native GPU architecture if not specified
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
        set(CMAKE_CUDA_ARCHITECTURES "native")
    else()
        # Adjust these values for your target GPUs:
        #   - 52 = Maxwell (GTX 9xx)
        #   - 75 = Turing (RTX 20xx, GTX 16xx)
        #   - 86 = Ampere (RTX 30xx)
        #   - 89 = Ada Lovelace (RTX 40xx)
        #   - 90 = Hopper (High-end / datacenter GPUs)

        set(CMAKE_CUDA_ARCHITECTURES 75 86 89)
    endif()
endif()

# ==================================================================
# Raylib Dependency
# ==================================================================

# If RAYLIB_DIR is provided: use that source tree
# Otherwise: Automatically download and build raylib
set(RAYLIB_DIR "" CACHE PATH
    "Path to a local raylib source tree (with CMakeLists.txt). If empty, raylib will be fetched from GitHub."
)

set(RAYLIB_VERSION "5.5" CACHE STRING
    "Raylib version to fetch from GitHub (ignored if RAYLIB_DIR is set)"
)

# raylib options: Don't build examples/games to boost the speed of raylib setup
set(BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(BUILD_GAMES    OFF CACHE BOOL "" FORCE)

if(RAYLIB_DIR)
    # Use provided raylib source
    if(EXISTS "${RAYLIB_DIR}/CMakeLists.txt")
        message(STATUS "Using local raylib from: ${RAYLIB_DIR}")
        add_subdirectory("${RAYLIB_DIR}" raylib_build)
        set(RAYLIB_IS_LOCAL TRUE)
    else()
        message(FATAL_ERROR
            "RAYLIB_DIR is set to '${RAYLIB_DIR}', but no CMakeLists.txt was found."
        )
    endif()
else()
    # No path provided: fetch raylib from GitHub
    include(FetchContent)
    set(FETCHCONTENT_QUIET OFF)
    
    message(STATUS "RAYLIB_DIR not set; Fetching raylib ${RAYLIB_VERSION} from GitHub...")

    FetchContent_Declare(
        raylib
        GIT_REPOSITORY https://github.com/raysan5/raylib.git
        GIT_TAG        ${RAYLIB_VERSION}
        GIT_SHALLOW    TRUE
    )
    FetchContent_MakeAvailable(raylib)
    set(RAYLIB_IS_FETCHED TRUE)
endif()

# ==================================================================
# raylib_cuda Library
# ==================================================================

add_library(raylib_cuda STATIC
    src/raylib_cuda.c
    src/raylib_cuda_backend.cu
)

# Add alias for modern CMake usage
add_library(raylib_cuda::raylib_cuda ALIAS raylib_cuda)

# Include directories
target_include_directories(
    raylib_cuda
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Link raylib (Handles OpenGL Linking)
target_link_libraries(raylib_cuda PUBLIC raylib)

# Handle static raylib on Windows
get_target_property(RAYLIB_TYPE raylib TYPE)
if(RAYLIB_TYPE STREQUAL "STATIC_LIBRARY")
    target_compile_definitions(raylib_cuda PUBLIC RAYLIB_STATIC)
endif()

# Language standards
set_target_properties(raylib_cuda PROPERTIES
    C_STANDARD 99
    C_STANDARD_REQUIRED ON
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED ON
    CUDA_STANDARD 11
    CUDA_STANDARD_REQUIRED ON
    POSITION_INDEPENDENT_CODE ON
)

# ==================================================================
# Platform-Specific Configuration
# ==================================================================

if(WIN32)
    # Link OpenGL explicitly for CUDA-GL interop functions
    target_link_libraries(raylib_cuda PRIVATE opengl32)
elseif(APPLE)
    message(WARNING "macOS support is experimental - CUDA may not work correctly")
    # macOS: Link OpenGL framework
    find_package(OpenGL REQUIRED)
    target_link_libraries(raylib_cuda PRIVATE OpenGL::GL)
else()
    # Linux: Link OpenGL
    find_package(OpenGL REQUIRED)
    target_link_libraries(raylib_cuda PRIVATE OpenGL::GL)
endif()

# ==================================================================
# Tests
# ==================================================================

if(RLC_BUILD_TESTS)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/tests/CMakeLists.txt")
        message(STATUS "Building tests...")
        add_subdirectory(tests)
    else()
        message(STATUS "RLC_BUILD_TESTS is ON but tests/ directory not found - skipping")
    endif()
endif()

# ==================================================================
# Examples
# ==================================================================

if(RLC_BUILD_EXAMPLES)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/examples/CMakeLists.txt")
        message(STATUS "Building examples...")
        add_subdirectory(examples)
    else()
        message(STATUS "RLC_BUILD_EXAMPLES is ON but examples/ directory not found - skipping")
    endif()
endif()

# ==================================================================
# Installation
# ==================================================================

include(GNUInstallDirs)

# Install library binary
install(TARGETS raylib_cuda
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

# Install headers
install(
    DIRECTORY include/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

# Note: Export targets are not created because raylib is fetched/built locally.
# Users should use add_subdirectory() or FetchContent to include raylib_cuda.
# Example usage in consuming projects:
#
#   FetchContent_Declare(raylib_cuda
#       GIT_REPOSITORY https://github.com/0bvdnt/raylib_cuda.git
#       GIT_TAG v1.1.0
#   )
#   FetchContent_MakeAvailable(raylib_cuda)
#   target_link_libraries(myapp PRIVATE raylib_cuda)

# ==================================================================
# Build Summary
# ==================================================================

message(STATUS "")
message(STATUS "=== raylib_cuda Configuration ===")
message(STATUS "   Version:             ${PROJECT_VERSION}")
message(STATUS "   CUDA Architectures:  ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "   Build Tests:         ${RLC_BUILD_TESTS}")
message(STATUS "   Build Examples:      ${RLC_BUILD_EXAMPLES}")
message(STATUS "   Raylib Version:      ${RAYLIB_VERSION}")
message(STATUS "   Install Prefix:      ${CMAKE_INSTALL_PREFIX}")

# Build type
get_property(IS_MULTI_CONFIG GLOBAL PROPERTY GENERATOR_IS_MULTI_CONFIG)
if(IS_MULTI_CONFIG)
    message(STATUS "   Generator:           ${CMAKE_GENERATOR} (multi-config)")
    message(STATUS "   Available Configs:   Debug, Release, RelWithDebInfo, MinSizeRel")
else()
    if(NOT CMAKE_BUILD_TYPE)
        set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type" FORCE)
    endif()
    message(STATUS "   Generator:           ${CMAKE_GENERATOR}")
    message(STATUS "   Build Type:          ${CMAKE_BUILD_TYPE}")
endif()

message(STATUS "=================================")
message(STATUS "")
